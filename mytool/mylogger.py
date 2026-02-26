import logging

def getLogger():
    # 创建一个logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('example.log')
    file_handler.setLevel(logging.DEBUG)  # 设置handler的日志级别为DEBUG

    # 创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 设置handler的日志级别为DEBUG

    # 创建一个formatter，设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 添加formatter到handler
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger



# from mytool.mylogger import getLogger
# logger = getLogger()
# # 现在使用logger记录信息
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')