class AppError(Exception):
    """面向用户的业务错误，message 可直接展示给使用者。"""

    def __init__(self, message: str, code: str = "APP_ERROR", detail: str | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail

    @property
    def user_message(self) -> str:
        if self.detail:
            return f"{self.message}\n{self.detail}"
        return self.message
