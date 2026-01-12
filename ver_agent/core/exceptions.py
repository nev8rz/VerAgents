class VerAgentException(Exception):
    pass


class LLMAgentException(VerAgentException):
    pass


class ConfigException(VerAgentException):
    pass


class AgentException(VerAgentException):
    pass


class ToolException(VerAgentException):
    pass


class MemoryException(VerAgentException):
    pass
