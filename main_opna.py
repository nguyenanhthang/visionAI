import App_NA_patmax
import threading
import plc_panasonic
import requests
import serial
from INI import Ini
from datetime import datetime
import os
import cv2

sys = Ini('sys_opna.ini')
sys.replace('True', 'enable_start')
COM_PLC = sys.find('PLC')
COM_LEFT = sys.find('SCAN_LEFT')
COM_RIGHT = sys.find('SCAN_RIGHT')
COM_MOTOR = sys.find('SAN_MOTOR')
check_on_off_sfc = sys.find('on_off_sfc')
sn_link1 = sys.find('sn_link1')
sn_link2 = sys.find('sn_link2')
url_post = sys.find('url_post')
url_save_img = sys.find('url_save_img')
staff_id = sys.find('staff_id')
enable_staff_id = sys.find('enable_start')
emp_link = sys.find('emp_link')
token_link = sys.find('token_link')

_AOI = App_NA_patmax.App()

def scan_staff():
    while True:
        if ser_motor.isOpen():
            ser_motor.write(bytearray(b'\x01\x54\x04'))
        else:
            ser_motor.open()
            ser_motor.write(bytearray(b'\x01\x54\x04'))
        data = ser_motor.readline(100)
        data_scan = data.decode()
        value_product_id = data_scan.strip()
        if len(value_product_id) > 0:
            ser_motor.close()
            value_staff_id = value_product_id
            if len(value_staff_id) == 8:
                data_staff_id = value_staff_id
                emp_link_staff_id = emp_link + data_staff_id
                req_token = requests.get(token_link)
                token_data = req_token.json()['data']['token']
                req_emp = requests.get(emp_link_staff_id,headers={'token':token_data})
                if req_emp.status_code == 200:
                    # _AOI.btn_nof_sfc.configure(text = "Connect Successfully!",text_color=("white"))
                    if req_emp.json()['data'] != 'null':
                        # staff_id = req_emp.json()['data']['staffCode']
                        # staff_name = req_emp.json()['data']['staffName']
                        # _AOI.btn_name_staffid.configure(text = str(staff_id))
                        # sys.replace(staff_id, 'staff_id')
                        # return True
                        try:
                            staff_id = req_emp.json()['data']['staffCode']
                            staff_name = req_emp.json()['data']['staffName']
                            date_job = req_emp.json()['data']['job']
                            _,d,m,y=date_job.split('/')
                            time_now = datetime.now().strftime("%Y/%m/%d")
                            date_job = f"{y}/{m}/{d}"
                            print(date_job)
                            print(req_emp.json()['data'])
                            if date_job <= time_now:
                                _AOI.btn_nof.configure(text= f"Mã nhân viên {staff_id} đã hết hạn {d}/{m}/{y}", text_color = "red")
                            else:
                                _AOI.btn_nof.configure(text= "", text_color = "")
                                _AOI.btn_nof_sfc.configure(text = "Connect Successfully!",text_color=("white"))
                                sys.replace(date_job, 'dateEx')
                                _AOI.btn_name_staffid.configure(text = str(staff_id) + f' - Hạn: {d}/{m}/{y}', text_color = 'white')
                                sys.replace(staff_id, 'staff_id')
                                return True
                        except:
                            print(req_emp.json()['data'])
                            _AOI.btn_nof.configure(text= "Không tìm thấy ngày hết hạn chứng chỉ!", text_color = "red")
                    else:
                        _AOI.btn_nof.configure(text = "Ma nhan vien khong ton tai!", text_color=("red"))
            else:
                _AOI.btn_name_staffid.configure(text = "You not Employee!",text_color=("red"))
# def check_same(material):
#     path = "check_same.txt"
#     if os.path.exists(path): 
#         with open(path, "r") as f:
#             if any(material == i.strip() for i in f.readlines()):
#                 # btn_L_R.configure(text = 'Trung ma lieu!')
#                 _AOI.btn_nof.configure(text= "Trung ma lieu!!, vui lòng quét lại", text_color = "red")
#                 return False
#     with open(path, "a", encoding="utf-8") as w:
#         w.write(material)
#         w.write("\n")
#     return True
def scan_material_screws():
    name_material = sys.find("name_material")
    name_material = name_material.replace(" ", "")
    name_material = name_material.split(",")
    same_material = open('check_material.ini', 'r')
    same_material = same_material.readlines()
    ser_motor = serial.Serial(COM_MOTOR, 9600, 8, timeout = 1)
    sys.replace('False', 're_scan_material')
    while True:
        if ser_motor.isOpen():
            ser_motor.write(bytearray(b'\x01\x54\x04'))
        else:
            ser_motor.open()
            ser_motor.write(bytearray(b'\x01\x54\x04'))
        data = ser_motor.readline(150)
        data_scan = str(data)[14:-3]
        for x in name_material:
            if len(data_scan) > 0:
                for same in same_material:
                    same = same.strip()
                    if data_scan == same:
                        _AOI.btn_nof.configure(text = 'Same material')
                        ser_motor.close()
                        return
                if data_scan.__contains__(x):
                    try:
                        a = data_scan.split(",")
                        if a[2][0] == "Q":
                            print(a[2][2:])
                            cv_ok = float(a[2][2:])
                            nb = int(cv_ok)
                            _AOI.btn_remainning_screws.configure(text = str(nb))
                            _AOI.btn_quantity_screws.configure(text = str(nb))
                            sys.replace(str(nb), 'remainning_screws')
                            sys.replace(str(nb), 'quantity_screws')
                            new_data = data_scan.replace(';', ',')
                            sys.replace(str(new_data), 'url_material')
                            _AOI.btn_nof.configure(text = '')
                            ser_motor.close()
                            sys.replace('True', 're_scan_material')
                            with open('check_material.ini', 'a') as f:
                                f.write("\n" + str(data_scan))
                            return True
                            
                    except:
                        _AOI.btn_nof.configure(text= "không tìm thấy số lượng Screws!", text_color = "red")
                else:
                    _AOI.btn_nof.configure(text= "Sai mã Screws!, vui lòng quét lại", text_color = "red")
                
def post_data():
    try:
        url_material = sys.find('url_material')
        # moment1 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6000'))/1000
        # moment2 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6001'))/1000
        # moment3 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6002'))/1000
        time_now = datetime.now()
        text_file = time_now.strftime("%Y%m%d")    
        path_file_seve = '//10.222.48.222/SensorAOI/OPALNA-ACTUATOR/'+ text_file
        if not os.path.exists(path_file_seve):
            os.makedirs(path_file_seve)
        
        value_product_replace = value_product.replace(':','_')
        text_time=time_now.strftime("_%Y.%m.%d %H.%M.%S_")
        img_print_RGB = cv2.cvtColor(_AOI.img, cv2.COLOR_BGR2RGB)
        if _AOI.result == 1:
            cv2.imwrite(url_save_img + str(text_file) + "/" + str(value_product_replace) + str(text_time) + "Passed.png",img_print_RGB)
            data_post = {"sn":value_product,
                "torque1":moment1,
                "torque2":moment2,
                "torque3":moment3,
                "keypart":value_motor,
                "stationName":"OPALNA-ACTUATOR",
                "empNo":staff_id,
                "result":"PASS",
                "material1":url_material
                }
            print(data_post)
            req_post = requests.post(url_post,json = data_post)
            if req_post.text.__contains__('200'):
                _AOI.btn_nof_sfc.configure(text = 'Sucessfully post data',text_color=("white"))
            else:
                _AOI.btn_nof_sfc.configure(text = f'Fail {req_post.text}',text_color=("red"))
                plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',2)
                if _AOI.Lock_QC():
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',1)
        else:
            cv2.imwrite(url_save_img + str(text_file) + "/" + str(value_product_replace) + str(text_time) + "Failed.png",img_print_RGB)
            data_post = {"sn":value_product,
                "torque1":moment1,
                "torque2":moment2,
                "torque3":moment3,
                "keypart":value_motor,
                "stationName":"OPALNA-ACTUATOR",
                "empNo":staff_id,
                "result":"FAIL",
                "material1":url_material
                }
            print(data_post)
            req_post = requests.post(url_post,json = data_post)
            if req_post.text.__contains__('200'):
                _AOI.btn_nof_sfc.configure(text = 'Sucessfully post data',text_color=("white"))
            else:
                _AOI.btn_nof_sfc.configure(text = f'Fail {req_post.text}',text_color=("red"))
    except Exception as ex:
        _AOI.btn_nof.configure(text = 'Error post data: ' + str(ex))

def main():
    global value_motor, value_product, ser_motor, moment1, moment2, moment3
    ser_R = serial.Serial(COM_RIGHT, 115200, 8, timeout = 1)
    ser_L = serial.Serial(COM_LEFT, 115200, 8, timeout = 1)
    ser_motor = serial.Serial(COM_MOTOR, 9600, 8, timeout = 1)
    while True:
        try:
            on_off_material = sys.find('Material')
            re_screws = int(sys.find('remainning_screws'))
            enable_staff_id = sys.find('enable_start')
            check_on_off_sfc = sys.find('on_off_sfc')
            product_sp1 = sys.find('product_sample1')
            product_sp2 = sys.find('product_sample2')
            product_sp3 = sys.find('product_sample3')
            if enable_staff_id == True:
                _AOI.btn_name_staffid.configure(text = 'Scan Employee', text_color = 'red')
                scan_staff()
                sys.replace('False', 'enable_start')
                _AOI.btn_name_staffid.configure(text_color = 'white')

            if on_off_material == True:
                if re_screws <= 0:
                    _AOI.btn_nof.configure(text = 'Hết liệu Screws!', text_color = 'red')
                    scan_material_screws()
                    re_screws = int(sys.find('remainning_screws'))

            read_plc_scan_right = plc_panasonic.read_data_panasonic(COM_PLC, 9600, 'D17890')
            read_plc_scan_left = plc_panasonic.read_data_panasonic(COM_PLC, 9600, 'D17893')
            read_plc_scan_test = plc_panasonic.read_data_panasonic(COM_PLC, 9600, 'D18890')
            startleftPLC = plc_panasonic.read_data_panasonic(COM_PLC, 9600, 'D17803')
            startrightPLC = plc_panasonic.read_data_panasonic(COM_PLC, 9600, 'D17800')
            read_plc_aoi_test = plc_panasonic.read_data_panasonic(COM_PLC, 9600, 'D18803')

            if enable_staff_id == False:
                if read_plc_scan_right == '1':
                    sys.replace('False', 'Test_sample')
                    print('__da nhan tin hieu start Scan phai__')
                    _AOI.btn_product_id.configure(text = '')
                    _AOI.btn_motor_id.configure(text = '')
                    _AOI.btn_nof.configure(text = '')
                    _AOI.btn_nof_sfc.configure(text = '')
                    _AOI.lbl_L_R.configure(text = 'RIGHT')
                    _AOI.canvas_R.delete('all')
                    
                    if check_on_off_sfc == True:
                        # ser_motor.reset_input_buffer()
                        if ser_R.isOpen():
                            ser_R.write(bytearray(b'\x02\xF4\x03'))
                        else:
                            ser_R.open()
                            ser_R.write(bytearray(b'\x02\xF4\x03'))

                        if ser_motor.isOpen():
                            ser_motor.write(bytearray(b'\x01\x54\x04'))
                        else:
                            ser_motor.open()
                            ser_motor.write(bytearray(b'\x01\x54\x04'))

                        data_product = ser_R.readline(29)
                        data_scan_product = str(data_product)[2:31]
                        value_product_id = data_scan_product
                        # ser_motor.reset_input_buffer()
                        data_mor = ser_motor.readline(35)
                        data_mor = data_mor.decode().strip()

                        if len(value_product_id) == 29:
                            _AOI.btn_product_id.configure(text = value_product_id,text_color=("white"))
                            
                            if len(data_mor) >= 25 and data_mor[0] == 'P':
                                ser_R.close()
                                ser_motor.close()
                                value_product = value_product_id
                                value_motor = data_mor
                                _AOI.btn_motor_id.configure(text = value_motor,text_color=("white"))
                                motor_link = sn_link1 + value_product + "&part_no=" + value_motor + sn_link2
                                req_motor_link = requests.get(motor_link)
                                if req_motor_link.text == '0':
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17892', 1)
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17890',0)
                                else:
                                    _AOI.btn_nof_sfc.configure(text = str(req_motor_link.text), text_color = 'red')
                    else:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17892', 1)
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17890',0)

                if startrightPLC == '1':
                    print('__da nhan tin hieu start AOI phai__')
                    try:
                        moment1 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6000'))/1000
                        moment2 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6001'))/1000
                        moment3 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6002'))/1000
                    except:
                        print('loi moment')

                    _AOI.display_canvas_R(moment1, moment2, moment3)
                    if _AOI.result == 1:
                        if check_on_off_sfc == True:
                            post_data()
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17810',1)
                    else:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17810',2)
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',2)
                        if check_on_off_sfc == True:
                            post_data()
                        if _AOI.Lock_QC():
                            plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',1)
                    if on_off_material == True:
                        re_screws = re_screws - 3
                        sys.replace(str(re_screws), 'remainning_screws')
                        _AOI.btn_remainning_screws.configure(text = str(re_screws))
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17892',0)
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17800',0)
                                
                if read_plc_scan_left == '1':
                    sys.replace('False', 'Test_sample')
                    print('__da nhan tin hieu start Scan trai')
                    _AOI.lbl_L_R.configure(text = 'LEFT')
                    _AOI.btn_nof.configure(text = '')
                    _AOI.btn_nof_sfc.configure(text = '')
                    _AOI.canvas_L.delete('all')
                    
                    if check_on_off_sfc == True:
                        # ser_motor.reset_input_buffer()
                        if ser_L.isOpen():
                            ser_L.write(bytearray(b'\x02\xF4\x03'))
                        else:
                            ser_L.open()
                            ser_L.write(bytearray(b'\x02\xF4\x03'))

                        if ser_motor.isOpen():
                            ser_motor.write(bytearray(b'\x01\x54\x04'))
                        else:
                            ser_motor.open()
                            ser_motor.write(bytearray(b'\x01\x54\x04'))

                        data_product = ser_L.readline(29)
                        data_scan_product = str(data_product)[2:31]
                        value_product_id = data_scan_product
                        # ser_motor.reset_input_buffer()
                        data_mor = ser_motor.readline(35)
                        data_mor = data_mor.decode().strip()

                        if len(value_product_id) == 29:
                            _AOI.btn_product_id.configure(text = value_product_id,text_color=("white"))
                            
                            if len(data_mor) >= 25 and data_mor[0] == 'P':
                                ser_L.close()
                                ser_motor.close()
                                value_product = value_product_id
                                value_motor = data_mor
                                _AOI.btn_motor_id.configure(text = value_motor,text_color=("white"))
                                motor_link = sn_link1 + value_product + "&part_no=" + value_motor + sn_link2
                                req_motor_link = requests.get(motor_link)
                                if req_motor_link.text == '0':
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17895',1)
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17893',0)
                                else:
                                    _AOI.btn_nof_sfc.configure(text = str(req_motor_link.text), text_color = 'red')
                    else:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17895',1)
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17893',0)

                if startleftPLC == '1':
                    print('__da nhan tin hieu start AOI trai')
                    try:
                        moment1 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6000'))/1000
                        moment2 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6001'))/1000
                        moment3 = int(plc_panasonic.read_data_panasonic(COM_PLC,9600,'D6002'))/1000
                    except:
                        print('loi moment')

                    _AOI.display_canvas_L(moment1, moment2, moment3)
                    if _AOI.result == 1:
                        if check_on_off_sfc == True:
                            post_data()
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17810',1)
                    else:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17810',2)
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',2)
                        if check_on_off_sfc == True:
                            post_data()
                        if _AOI.Lock_QC():
                            plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',1)
                    if on_off_material == True:
                        re_screws = re_screws - 3
                        sys.replace(str(re_screws), 'remainning_screws')
                        _AOI.btn_remainning_screws.configure(text = str(re_screws))
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17895',0)
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17803',0)
                        
                if read_plc_scan_test == '1':
                    _AOI.lbl_L_R.configure(text = 'TEST SAMPLE')
                    sys.replace('False', 'Test_sample')
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18891',0)
                    if check_on_off_sfc == True:
                        if ser_R.isOpen():
                            ser_R.write(bytearray(b'\x02\xF4\x03'))
                        else:
                            ser_R.open()
                            ser_R.write(bytearray(b'\x02\xF4\x03'))

                        if ser_motor.isOpen():
                            ser_motor.write(bytearray(b'\x01\x54\x04'))
                        else:
                            ser_motor.open()
                            ser_motor.write(bytearray(b'\x01\x54\x04'))

                        data_product = ser_R.readline(29)
                        data_scan_product = str(data_product)[2:31]
                        value_product_id = data_scan_product
                        # ser_motor.reset_input_buffer()
                        data_mor = ser_motor.readline(35)
                        data_mor = data_mor.decode().strip()

                        if len(value_product_id) == 29:
                            _AOI.btn_product_id.configure(text = value_product_id,text_color=("white"))
                            if value_product_id == product_sp1 or value_product_id == product_sp2 or value_product_id == product_sp3:
                                sys.replace('True', 'Test_sample')
                            if len(data_mor) >= 25 and data_mor[0] == 'P':
                                ser_R.close()
                                ser_motor.close()
                                value_product = value_product_id
                                value_motor = data_mor
                                _AOI.btn_motor_id.configure(text = value_motor,text_color=("white"))
                                motor_link = sn_link1 + value_product + "&part_no=" + value_motor + sn_link2
                                req_motor_link = requests.get(motor_link)
                                if req_motor_link.text == '0':
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18891',1)
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18890',0)
                                else:
                                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18892',1)
                                    _AOI.btn_nof_sfc.configure(text = str(req_motor_link.text), text_color = 'red')
                    else:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18891',1)

                if read_plc_aoi_test == '1':
                    moment1 = 0.5
                    moment2 = 0.5
                    moment3 = 0.5
                    _AOI.display_canvas_T(moment1, moment2, moment3)
                    if _AOI.result == 1:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18810',1)
                        if check_on_off_sfc == True:
                            post_data()
                    else:
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18810',2)
                        plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',2)
                        if check_on_off_sfc == True:
                            post_data()
                        if _AOI.Lock_QC():
                            plc_panasonic.write_data_panasonic(COM_PLC,9600,'D17820',1)
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18891',0)
                    plc_panasonic.write_data_panasonic(COM_PLC,9600,'D18803',0)

        except Exception as ex:
            _AOI.btn_nof.configure(text = 'Error main: ' + str(ex))
            try:
                ser_L.close()
                ser_R.close()
                ser_motor.close()
            except:
                try:
                    ser_motor.close()
                except:
                    try:
                        ser_L.close()
                    except:
                        try:
                            ser_R.close()
                        except:
                            pass

if __name__ == '__main__':
    r1 = threading.Thread(daemon= True, target= main)
    r1.start()
    # r2 = threading.Thread(daemon= True, target=_AOI.windows)
    # r2.start()
    _AOI.mainloop()