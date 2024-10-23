# ==========================================================================
# Copyright (c) 2012-2024 Anonymous Authors

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==========================================================================

from logging import DEBUG, Formatter, StreamHandler, getLogger


def get_my_logger(module_name: str, level=DEBUG):
    # Set module name
    logger = getLogger(module_name)

    # Define log level
    logger.setLevel(level)

    # Define format
    formatter = Formatter(
        "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]: %(message)s")

    # Define hander for stout
    stream_handler = StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # # Define handler for files
    # logs_dir_name = "logs"
    # os.makedirs(logs_dir_name, exist_ok=True)
    # dt_now_jst_aware = datetime.datetime.now(
    #     datetime.timezone(datetime.timedelta(hours=9)))
    # file_name = dt_now_jst_aware.strftime("%Y%m%d_%H%M%S") + ".log"
    # file_path = os.path.join(logs_dir_name, file_name)
    # file_handler = FileHandler(file_path)
    # file_handler.setLevel(LEVEL)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger
