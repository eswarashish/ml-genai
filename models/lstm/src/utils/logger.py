import logging,sys
from logging import Logger
from typing import Optional
from uvicorn.logging import DefaultFormatter
apilogger  = logging.getLogger("uvicorn.error")



def getLogger(name: Optional[str])->Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        log_format = DefaultFormatter(fmt='%(levelprefix)s %(asctime)s - %(name)s - (%(filename)s:%(lineno)d) - %(message)s',
                                    use_colors=True)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    return logger
