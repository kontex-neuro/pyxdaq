import logging

import colorlog

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter('%(log_color)s%(asctime)s [%(levelname)s] %(name)s %(message)s')
)
logging.basicConfig(handlers=[handler], level=logging.INFO)
