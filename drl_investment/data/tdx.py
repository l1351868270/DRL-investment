
from datetime import datetime
from struct import unpack

def unpack_data(file_path, begin_time: datetime=None, end_time: datetime=None):
    '''
        https://blog.csdn.net/m0_46603114/article/details/112756894
    '''

    with open(file_path, 'rb') as source_file
        buf = source_file.read()
        source_file.close()
        buf_size = len(buf)
        rec_count = int(buf_size / 32)
        
        begin = buf_size - 32
        end = buf_size
        data = []
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