from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition

class OurTimeoutTerminal(TimeoutCondition):
    def __init__(self):
        super().__init__()
    
    def getStep():
        return super().steps