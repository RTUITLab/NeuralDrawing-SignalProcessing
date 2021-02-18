import queue
from threading import Thread
import numpy as np
import cyPyWinUSB as hid
from cyCrypto.Cipher import AES
import Model
from socket import *
import sys
import pandas as pd
import time


def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


class FileEEG(object):
    def __init__(self, filename):
        self.data_queue = queue.Queue()
        self.data = pd.read_csv(filename).values
        self.start()

    def get_data(self):
        return self.data_queue.get()

    def start(self):
        self.is_reading = True
        self.reading_thread = Thread(target=lambda: self.__start_reading())
        self.reading_thread.start()

    def stop(self):
        self.is_reading = False

    def __start_reading(self):
        read_count = 0  # Общее число прочитанных записей

        # Номер последней прочитанной записи;
        # при достижении конца массива данных сбрасывается до 0
        last_index = 0

        start_time = time.time()  # Время начала считывания
        frequency = 128
        count_between_reading = 20
        seconds_between_reading = count_between_reading / frequency
        while self.is_reading:
            current_time = time.time()
            if ((current_time - start_time) /
                seconds_between_reading) * count_between_reading > read_count:
                self.data_queue.put(list(self.data[last_index][2:]))
                last_index += 1
                if last_index >= len(self.data):
                    last_index = 0
                read_count += 1
            else:
                time.sleep(seconds_between_reading / 2)


class EEG(object):
    def __init__(self):
        self.hid = None
        self.delimiter = ","
        self.data_queue = queue.Queue()
        devicesUsed = 0

        for device in hid.find_all_hid_devices():
            if device.product_name == 'EEG Signals':
                devicesUsed += 1
                self.hid = device
                self.hid.open()
                self.serial_number = device.serial_number
                device.set_raw_data_handler(self.dataHandler)
        if devicesUsed == 0:
            raise RuntimeError('Устройство не найдено!')

        sn = self.serial_number

        # EPOC+ in 16-bit Mode.
        k = ['\0'] * 16
        k = [sn[-1], sn[-2], sn[-2], sn[-3], sn[-3], sn[-3], sn[-2], sn[-4], sn[-1], sn[-4], sn[-2], sn[-2], sn[-4],
             sn[-4], sn[-2], sn[-1]]

        # # EPOC+ in 14-bit Mode.
        # k = [sn[-1],00,sn[-2],21,sn[-3],00,sn[-4],12,sn[-3],00,sn[-2],68,sn[-1],00,sn[-2],88]

        self.key = str(''.join(k))
        self.cipher = AES.new(self.key.encode("utf8"), AES.MODE_ECB)

    def dataHandler(self, data):
        join_data = ''.join(map(chr, data[1:]))
        data = self.cipher.decrypt(bytes(join_data, 'latin-1')[0:32])
        if str(data[1]) == "32":  # No Gyro Data.
            return
        try:
            packet_data = []
            for i in range(2, 16, 2):
                packet_data.append(float(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1]))))

            for i in range(18, len(data), 2):
                packet_data.append(float(self.convertEPOC_PLUS(str(data[i]), str(data[i + 1]))))
            # packet_data = packet_data[:-len(self.delimiter)]
            self.data_queue.put(packet_data)
            return str(packet_data)

        except Exception as exception:
            print(str(exception))

    def convertEPOC_PLUS(self, value_1, value_2):
        edk_value = "%.8f" % (((int(value_1) * .128205128205129)
                               + 4201.02564096001)
                              + ((int(value_2) - 128) * 32.82051289))
        return edk_value

    def get_data(self):
        return self.data_queue.get()


class SignalProcessor(object):
    HEADSET_FREQUENCY = 128  # Гарнитура выдает 128 значений в секунду

    def __init__(self, model_name=None, read_from_file=False, host=None, port=None):
        # choose reading variant
        if read_from_file:
            self.eeg = FileEEG('concentrate_t.csv')
        else:
            self.eeg = EEG()

        self.host = host
        self.port = port

        self.model = None
        self.current_data = []
        self.is_predicting = False

        if model_name is not None:
            self.load_model(model_name)

    def load_model(self, model_name):
        self.model = Model.Model('best.pth', ['concentrate', 'relax'])

    def get_data_queue(self):
        return self.eeg.data_queue

    def clear_data_queue(self):
        while not self.eeg.data_queue.empty():
            self.eeg.data_queue.get()

    def start_predicting(self, interval_sec=1):
        if self.model is None:
            raise RuntimeError("Невозможно начать предсказывать результат: "
                               + "модель не задана")
        if self.is_predicting:
            return

        self.clear_data_queue()
        self.is_predicting = True
        self.thread = Thread(target=lambda: self.__predicting(interval_sec))
        self.thread.start()

    def stop_predicting(self):
        self.is_predicting = False

    global counter
    global last_values

    counter = 0
    last_values = np.zeros(30)

    def value_predicted(self, value):
        '''Метод вызывается каждый раз,
        когда моделью было предсказано значение value
        '''

        host = self.host
        port = self.port
        addr = (host, port)

        udp_socket = socket(AF_INET, SOCK_DGRAM)
        # encode - перекодирует введенные данные в байты, decode - обратно

        global counter
        print(counter > 29, counter)
        if counter > 29:
            shift5(last_values, 1)
            last_values[-((counter + 1) % 30)] = value
            counter += 1
            data = str(last_values.mean())
            data = str.encode(data)
            udp_socket.sendto(data, addr)
            data = bytes.decode(data)
            data = udp_socket.recvfrom(1024)
        last_values[counter] = value
        data = str(last_values.mean())
        counter += 1
        data = str.encode(data)
        udp_socket.sendto(data, addr)
        data = bytes.decode(data)
        data = udp_socket.recvfrom(1024)
        udp_socket.close()

    def __predicting(self, interval_sec):
        count_between_predicts = int(self.HEADSET_FREQUENCY * interval_sec)
        model_required_count = self.HEADSET_FREQUENCY * 1  # будет зависеть от модели
        while self.is_predicting:
            if len(self.current_data) < model_required_count:
                self.current_data.append(self.eeg.get_data())
            else:
                self.value_predicted(
                    self.model.process_data(np.array(self.current_data)))
                for _ in range(count_between_predicts):
                    self.current_data.pop(0)


if __name__ == '__main__':
    sp = SignalProcessor('best.pth', read_from_file=True)
    sp.start_predicting(0.1)
