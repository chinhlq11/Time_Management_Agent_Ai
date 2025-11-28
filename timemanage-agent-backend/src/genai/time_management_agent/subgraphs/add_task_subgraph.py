from langgraph.graph import StateGraph, END
from ..nodes.subgraph.add_task_nodes import add_task_node
from ..nodes.reset_topic import reset_topic_node
from ..states.time_management_agent_state import TimanaAgentState

def build_add_task_subgraph():
    graph = StateGraph(TimanaAgentState)
    graph.add_node("add_task_node", add_task_node)
    graph.add_node("reset_topic_node", reset_topic_node)

    graph.set_entry_point("add_task_node") 
    graph.add_edge("add_task_node", "reset_topic_node") 
    graph.add_edge("reset_topic_node", END) 

    return graph.compile()