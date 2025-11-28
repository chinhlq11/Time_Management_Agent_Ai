from ..states.time_management_agent_state import TimanaAgentState
from logger import logger

TOPIC_FLOW_MAPPING = {
    "greeting": "greeting",
    "off_topic": "off_topic",
    "add_task":"add_task",
    "get_tasks":"get_tasks"
}


def flow_controller_node(state: TimanaAgentState):
    selected_flow = TOPIC_FLOW_MAPPING[state['topic'].name]
    logger.info(f"Selected Flow: {selected_flow}")

    return {
        "selected_flow": selected_flow
    }

def get_selected_flow(state: TimanaAgentState):
    return state.get('selected_flow', TOPIC_FLOW_MAPPING['off_topic'])