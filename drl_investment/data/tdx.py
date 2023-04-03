
from datetime import datetime
import logging
from struct import unpack


LOG = logging.getLogger(__name__)


def unpack_data(file_path, begin_time: datetime=None, end_time: datetime=None):
    '''
    The columns of the file is that
        0: 'date'
        1: 'open'
        2: 'high'
        3: 'low'
        4: 'close'
        5: 'amount'
        6: 'volume'
    Example:
        [datetime.datetime(2022, 12, 5, 0, 0), '3181.92', '3213.44', '3177.06', '3211.81', '479203622912.0', '447398437']
    reference:https://blog.csdn.net/m0_46603114/article/details/112756894
    '''

    with open(file_path, 'rb') as source_file:
        buf = source_file.read()
        source_file.close()
        buf_size = len(buf)
        rec_count = int(buf_size / 32)
        
        begin = buf_size - 32
        end = buf_size
        data = []
        LOG.debug(f'The total lines of the file {file_path} is {rec_count}')
        for i in range(rec_count):
            a = unpack('IIIIIfII', buf[begin:end])
            
            year = a[0] // 10000
            month = (a[0] % 10000) // 100
            day = (a[0] % 10000) % 100
            date = datetime.fromisoformat('{}-{:02d}-{:02d}'.format(year, month, day))

            if begin_time is not None:
                if date < begin_time:
                    continue
            if end_time is not None:
                if date > end_time:
                    continue
            data.append([date, str(a[1] / 100.0), str(a[2] / 100.0), str(a[3] / 100.0), \
                        str(a[4] / 100.0), str(a[5]), str(a[6])])
            begin -= 32
            end -= 32

        data.reverse()
        return data