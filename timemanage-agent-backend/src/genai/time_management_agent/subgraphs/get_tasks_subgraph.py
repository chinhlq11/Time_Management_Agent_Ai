from langgraph.graph import StateGraph, END
from ..nodes.subgraph.get_tasks_nodes import get_tasks_node
from ..nodes.reset_topic import reset_topic_node
from ..states.time_management_agent_state import TimanaAgentState

def build_get_tasks_subgraph():
    graph = StateGraph(TimanaAgentState)
    graph.add_node("get_tasks_node", get_tasks_node)
    graph.add_node("reset_topic_node", reset_topic_node)

    graph.set_entry_point("get_tasks_node") # START -> get_tasks_node
    graph.add_edge("get_tasks_node", "reset_topic_node") # get_tasks_node -> reset_topic_node
    graph.add_edge("reset_topic_node", END) # reset_topic_node -> END

    return graph.compile()