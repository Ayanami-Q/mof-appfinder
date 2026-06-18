from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from mof_agent.state import AgentState
from mof_agent.nodes import (
    node_analyze, node_think_strategy, node_act_retrieve,
    node_review_candidates, node_fetch_literature, node_final_report,
)


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("node_analyze", node_analyze)
    builder.add_node("node_think_strategy", node_think_strategy)
    builder.add_node("node_act_retrieve", node_act_retrieve)
    builder.add_node("node_review_candidates", node_review_candidates)
    builder.add_node("node_fetch_literature", node_fetch_literature)
    builder.add_node("node_final_report", node_final_report)

    builder.add_edge(START, "node_analyze")
    builder.add_edge("node_analyze", "node_think_strategy")
    builder.add_edge("node_think_strategy", "node_act_retrieve")
    builder.add_edge("node_act_retrieve", "node_review_candidates")
    builder.add_edge("node_review_candidates", "node_fetch_literature")
    builder.add_edge("node_fetch_literature", "node_final_report")
    builder.add_edge("node_final_report", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
