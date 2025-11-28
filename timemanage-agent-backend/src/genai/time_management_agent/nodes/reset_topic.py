from ..states.time_management_agent_state import TimanaAgentState

def reset_topic_node(state: TimanaAgentState):
    return {
        "topic": None
    }