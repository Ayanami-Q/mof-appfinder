import os
import sys
import re
import io
import shutil
import tempfile
from typing import Dict, Any, Optional
from xml.etree import ElementTree as ET

import requests

from mof_agent.config import PUBLISHER_API_KEYS, PUBLISHER_API_ENABLED

_publisher_fetcher = None

PDF_SOURCES = [
    "https://api.unpaywall.org/v2/{doi}?email=dev@mof-agent.org",
    "https://sci-hub.se/{doi}",
    "https://sci-hub.ru/{doi}",
    "https://sci-hub.st/{doi}",
]
PDF_DOWNLOAD_TIMEOUT = 10


def _get_publisher_fetcher():
    global _publisher_fetcher
    if _publisher_fetcher is None and PUBLISHER_API_ENABLED:
        try:
            # text_get is an optional external module
            _tools_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, _tools_dir)
            from text_get import AcademicPaperFetcher
            _publisher_fetcher = AcademicPaperFetcher(
                elsevier_key=PUBLISHER_API_KEYS["elsevier"],
                springer_key=PUBLISHER_API_KEYS["springer"],
                wiley_token=PUBLISHER_API_KEYS["wiley"],
            )
            print("[LitFetch] Publisher API fetcher ready (Elsevier/Springer/Wiley)")
        except ImportError:
            print("[LitFetch] Warning: tools/text_get.py not found, publisher API unavailable")
        except Exception as e:
            print(f"[LitFetch] Warning: publisher API init failed: {e}")
    return _publisher_fetcher


def _xml_to_markdown(xml_text: str, source: str = "elsevier") -> Optional[str]:
    try:
        root = ET.fromstring(xml_text)
        ns_map = {
            "ce": "http://www.elsevier.com/xml/common/dtd",
            "ja": "http://www.elsevier.com/xml/ja/dtd",
            "xocs": "http://www.elsevier.com/xml/xocs/dtd",
            "jats": "http://www.ncbi.nlm.nih.gov/JATS1",
        }
        parts = []
        title = None
        for xpath in [
            ".//ce:title", ".//jats:article-title", ".//*[local-name()='article-title']",
        ]:
            for ns, uri in ns_map.items():
                try:
                    el = root.find(xpath, {ns: uri})
                    if el is not None and el.text:
                        title = el.text.strip()
                        break
                except Exception:
                    pass
            if title:
                break
        if title:
            parts.append(f"# {title}\n")

        abstract_parts = []
        for xpath in [
            ".//ce:abstract", ".//jats:abstract", ".//*[local-name()='abstract']",
        ]:
            for ns, uri in ns_map.items():
                try:
                    for abs_el in root.findall(xpath, {ns: uri}):
                        text = "".join(abs_el.itertext()).strip()
                        if text:
                            abstract_parts.append(text)
                except Exception:
                    pass
        if abstract_parts:
            parts.append("## Abstract\n")
            parts.append("\n\n".join(abstract_parts[:2]))
            parts.append("")

        body_texts = []
        for tag in ["ce:para", "jats:p", "ce:section", "jats:sec"]:
            for ns, uri in ns_map.items():
                try:
                    for el in root.iter():
                        if el.tag.endswith("}" + tag.split(":")[-1]) or el.tag == tag:
                            text = "".join(el.itertext()).strip()
                            if text and len(text) > 80:
                                body_texts.append(text)
                except Exception:
                    pass

        if body_texts:
            max_chars = 5000
            current_len = sum(len(p) for p in parts)
            for t in body_texts:
                if current_len > max_chars:
                    parts.append("\n\n[... body text truncated ...]")
                    break
                parts.append(t)
                parts.append("")
                current_len += len(t)

        result = "\n\n".join(parts).strip()
        if len(result) > 200:
            return result
    except ET.ParseError:
        pass
    except Exception:
        pass
    return None


def _fetch_via_publisher_api(doi: str) -> Optional[str]:
    fetcher = _get_publisher_fetcher()
    if fetcher is None:
        return None
    try:
        raw = fetcher.get_paper(doi)
        if raw is None:
            return None
        if isinstance(raw, str):
            md = _xml_to_markdown(raw)
            if md:
                return md
            try:
                root = ET.fromstring(raw)
                text = "".join(root.itertext())
                if len(text) > 200:
                    return text[:6000]
            except Exception:
                pass
        elif isinstance(raw, bytes):
            if raw[:4] == b'%PDF':
                tmp_dir = tempfile.mkdtemp(prefix="pub_pdf_")
                try:
                    import PyPDF2
                    reader = PyPDF2.PdfReader(io.BytesIO(raw))
                    text = "\n".join(
                        page.extract_text() or "" for page in reader.pages[:10]
                    )
                    if text.strip():
                        return _trim_references(text)
                except Exception:
                    pass
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass
    return None


def _download_pdf_via_unpaywall(doi: str, timeout: int = PDF_DOWNLOAD_TIMEOUT) -> Optional[bytes]:
    url = f"https://api.unpaywall.org/v2/{doi}?email=dev@mof-agent.org"
    try:
        resp = requests.get(url, timeout=timeout,
                           headers={"User-Agent": "MOF-Agent/1.0"})
        if resp.status_code == 200:
            data = resp.json()
            oa_url = (data.get("best_oa_location") or {}).get("url_for_pdf")
            if oa_url:
                pdf_resp = requests.get(oa_url, timeout=timeout,
                                        headers={"User-Agent": "MOF-Agent/1.0"})
                if pdf_resp.status_code == 200 and len(pdf_resp.content) > 1000:
                    return pdf_resp.content
    except Exception:
        pass
    return None


def _download_pdf_direct(doi: str, timeout: int = PDF_DOWNLOAD_TIMEOUT) -> Optional[bytes]:
    for source_tpl in PDF_SOURCES[1:]:
        try:
            url = source_tpl.format(doi=doi)
            resp = requests.get(url, timeout=timeout, allow_redirects=True,
                               headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
                                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                                        "Chrome/120.0.0.0 Safari/537.36"})
            if resp.status_code == 200 and len(resp.content) > 1000:
                if resp.content[:4] == b'%PDF':
                    return resp.content
                if b'<html' in resp.content[:200].lower() or b'<!DOCTYPE' in resp.content[:200]:
                    pdf_urls = re.findall(r'(https?://[^"\']+\.pdf[^"\']*)',
                                         resp.content.decode('latin-1', errors='ignore'))
                    if pdf_urls:
                        pdf_resp = requests.get(pdf_urls[0], timeout=timeout,
                                               headers={"User-Agent": "Mozilla/5.0"})
                        if pdf_resp.status_code == 200 and pdf_resp.content[:4] == b'%PDF':
                            return pdf_resp.content
        except Exception:
            continue
    return None


def _parse_pdf_with_marker(pdf_content: bytes, tmp_dir: str) -> Optional[str]:
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        pdf_path = os.path.join(tmp_dir, "paper.pdf")
        with open(pdf_path, "wb") as f:
            f.write(pdf_content)

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(pdf_path)
        markdown, _, _ = text_from_rendered(rendered)
        return _trim_references(markdown)
    except Exception:
        pass

    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages[:8]
        )
        if text.strip():
            return _trim_references(text)
    except Exception:
        pass

    return None


def _trim_references(text: str) -> str:
    ref_markers = [
        r'\n#+\s*References?\s*\n', r'\n#+\s*Bibliography\s*\n',
        r'\n#+\s*REFERENCES\s*\n', r'\n\*\*References?\*\*\s*\n',
        r'\nReferences?\s*\n={3,}', r'\n\s*\[\d+\]\s',
    ]
    for marker in ref_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    max_chars = 8000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... content truncated ...]"
    return text.strip()


def _fetch_crossref_metadata_enhanced(doi: str, timeout: int = 10) -> Dict[str, str]:
    url = f"https://api.crossref.org/works/{doi}"
    result = {"doi": doi, "title": "", "authors": "", "abstract": "",
              "keywords": "", "journal": "", "year": ""}
    try:
        resp = requests.get(url, timeout=timeout,
                           headers={"User-Agent": "MOF-Agent/1.0 (mailto:dev@example.org)"})
        if resp.status_code != 200:
            return result
        data = resp.json()
        msg = data.get("message", {})
        result["title"] = msg.get("title", [""])[0] or ""
        result["abstract"] = (msg.get("abstract") or "")[:1200]
        author_list = msg.get("author", [])
        result["authors"] = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in author_list[:5]
        )
        if len(author_list) > 5:
            result["authors"] += " et al."
        result["journal"] = (msg.get("container-title", [""])[0] or "")
        result["year"] = str(
            msg.get("published-print", {}).get("date-parts", [[0]])[0][0] or
            msg.get("created", {}).get("date-parts", [[0]])[0][0] or ""
        )
        kw_list = msg.get("subject", [])
        result["keywords"] = ", ".join(kw_list[:8]) if kw_list else ""
    except Exception:
        pass
    return result


def _try_crossref_search(query: str, timeout: int = 5) -> Optional[str]:
    url = f"https://api.crossref.org/works?query={query}&rows=1"
    try:
        resp = requests.get(url, timeout=timeout,
                           headers={"User-Agent": "MOF-Agent/1.0 (mailto:dev@example.org)"})
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("message", {}).get("items", [])
            if items:
                title = items[0].get("title", ["N/A"])[0]
                doi = items[0].get("DOI", "N/A")
                abstract = items[0].get("abstract", "No abstract available.")
                return f"**{title}**\nDOI: {doi}\n{abstract}"
    except Exception:
        pass
    return None


class LitFetchTool:
    """Literature retrieval tool: wraps CrossRef, Unpaywall, and publisher API fetching."""

    @staticmethod
    def _format_kb_record(refcode: str, kb_entry: dict, header: str) -> str:
        """Format a knowledge base entry as Markdown literature content."""
        doi = kb_entry.get("doi", "")
        src = kb_entry.get("original_files", [])

        lines = [header]
        if doi:
            lines.append(f"DOI: [{doi}](https://doi.org/{doi})")
        if src:
            lines.append(f"**Source files**: {', '.join(src)}")
        lines.append("")

        fields = [
            ("synthesis_summary", "## Synthesis"),
            ("characterization_summary", "## Characterization"),
            ("applications_summary", "## Applications"),
            ("metal_valence", "## Metal Valence"),
            ("structural_description", "## Structure Description"),
            ("structure_activity_relationship", "## Structure-Activity Relationship"),
        ]
        for key, label in fields:
            val = kb_entry.get(key, "")
            if val and val not in ("Not provided", ""):
                lines.append(f"{label}\n{val}\n")

        just = kb_entry.get("justification", "")
        if just and just not in ("Not found in text", "Not provided", ""):
            lines.append(f"**Evidence**: {just}")

        return "\n".join(lines)

    @staticmethod
    def fetch(cif_id: str, physical_record: Optional[Dict] = None,
              match_score: Optional[Dict] = None) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "source": "heuristic",
            "content": "",
            "match_score": match_score or {},
        }
        header = f"### CoRE-MOF ID: {cif_id}"

        from mof_agent.context import get_csd_lookup
        csd = get_csd_lookup()
        if csd:
            doi = csd.get_doi(cif_id)
            name = csd.get_name(cif_id)
            refcode = csd.extract_refcode(cif_id)

            if refcode:
                header += f" | CSD Refcode: {refcode}"
            if name:
                header += f"\n**MOF Name**: {name}"

            # Prefer knowledge base
            if refcode:
                from mof_agent.context import get_knowledge_base
                kb = get_knowledge_base()
                if kb is not None:
                    kb_entry = kb.lookup(refcode)
                    if kb_entry is not None:
                        content = LitFetchTool._format_kb_record(refcode, kb_entry, header)
                        result["source"] = "knowledge_base"
                        result["content"] = content
                        return result

            if doi:
                pub_md = _fetch_via_publisher_api(doi)
                if pub_md and len(pub_md) > 100:
                    result["source"] = "publisher"
                    result["content"] = (
                        f"{header}\nDOI: [{doi}](https://doi.org/{doi})\n\n---\n{pub_md}\n"
                    )
                    return result

                tmp_dir = tempfile.mkdtemp(prefix="mof_lit_")
                try:
                    pdf_content = _download_pdf_via_unpaywall(doi)
                    if not pdf_content:
                        pdf_content = _download_pdf_direct(doi)
                    if pdf_content:
                        markdown = _parse_pdf_with_marker(pdf_content, tmp_dir)
                        if markdown and len(markdown) > 100:
                            result["source"] = "pdf"
                            result["content"] = (
                                f"{header}\nDOI: [{doi}](https://doi.org/{doi})\n\n---\n{markdown}\n"
                            )
                            shutil.rmtree(tmp_dir, ignore_errors=True)
                            return result
                except Exception:
                    pass
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

                meta = _fetch_crossref_metadata_enhanced(doi, timeout=10)
                if meta["title"] or meta["abstract"]:
                    result["source"] = "crossref"
                    lines = [header, f"DOI: [{doi}](https://doi.org/{doi})", ""]
                    if meta["title"]:
                        lines.append(f"**Title**: {meta['title']}")
                    if meta["authors"]:
                        lines.append(f"**Authors**: {meta['authors']}")
                    if meta["journal"]:
                        ji = f"{meta['journal']}"
                        if meta["year"]:
                            ji += f" ({meta['year']})"
                        lines.append(f"**Journal**: {ji}")
                    if meta["keywords"]:
                        lines.append(f"**Keywords**: {meta['keywords']}")
                    if meta["abstract"]:
                        lines.append(f"\n**Abstract**: {meta['abstract']}")
                    result["content"] = "\n".join(lines)
                    return result

            if refcode:
                search_result = _try_crossref_search(refcode, timeout=5)
                if search_result:
                    result["source"] = "crossref"
                    result["content"] = f"{header}\n{search_result}"
                    return result

        if physical_record:
            elems = ", ".join(physical_record.get("elements", ["?"]))
            lcd = physical_record.get("lcd", "?")
            pld = physical_record.get("pld", "?")
            vf = physical_record.get("vf", "?")
            vol = physical_record.get("volume", "?")

            if lcd and float(lcd) > 20:
                hint = "Large-pore material, suitable for macromolecular encapsulation, enzyme immobilization, or drug delivery."
            elif lcd and float(lcd) < 5:
                hint = "Microporous material, suitable for molecular sieving, gas separation (H2/CO2, C2H4/C2H6), or selective adsorption."
            elif vf and float(vf) > 0.7:
                hint = "High-porosity material, suitable for high-pressure gas storage (CH4, H2) or catalysis."
            else:
                hint = "Medium-porosity MOF, suitable for CO2 capture, aqueous pollutant adsorption, or fluorescence sensing."

            result["content"] = (
                f"{header}\n"
                f"**Elements**: {elems}\n"
                f"**Cell params**: a={physical_record.get('a')} A, b={physical_record.get('b')} A, "
                f"c={physical_record.get('c')} A, volume={vol} A3\n"
                f"**Pore features**: LCD={lcd} A, PLD={pld} A, VF={vf}\n"
                f"**Inferred direction**: {hint}\n"
            )
            return result

        result["content"] = f"{header}\n**Description**: This MOF is included in the CoRE-MOF database.\n"
        return result
