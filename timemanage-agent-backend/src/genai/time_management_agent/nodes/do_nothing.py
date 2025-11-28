from ..states.time_management_agent_state import TimanaAgentState
from logger import logger
def do_nothing_node(state: TimanaAgentState):
    logger.info("do_nothing_node called.")
    return {}