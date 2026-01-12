"""异常体系"""

class VerAgentsException(Exception):
    """VerAgents基础异常类"""
    pass

class LLMException(VerAgentsException):
    """LLM相关异常"""
    pass

class AgentException(VerAgentsException):
    """Agent相关异常"""
    pass

class ConfigException(VerAgentsException):
    """配置相关异常"""
    pass

class ToolException(VerAgentsException):
    """工具相关异常"""
    pass