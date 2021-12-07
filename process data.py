from sklearn.model_selection import train_test_split
import scipy.io as scio
import numpy as np
import os
import datetime
from glob import glob
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib as mpl
# mpl.use('TkAgg')
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import openpyxl
from openpyxl import Workbook

path_emotion_intensity = 'Database information(2).xlsx'
Path_cont = '/Contextual data/'
Path_eye = '/Eye data/'
Path_phy = '/Physical data/'
inputs_path = '/1data/'
labeled_list = ['labeled_segment5', 'labeled_segment7', 'labeled_segment9',
                'labeled_segment11', 'labeled_segment13']
time = 60 * 5  # 5s sliding window 10s
step = 60 * 5 * 0.1  # 90% Overlap ratio


def processdata(path_cont=Path_cont, path_eye=Path_eye, path_phy=Path_phy):
    print("Loaddata...")
    X_cont = []
    X_eye = []
    X_phy = []
    emotion_intensity = []
    #y_all 用于存放y的所有取值可能的二维数组，索引分别为场景，人
    y_all = [[ ],[ ],[ ],[ ],[ ]]
    y_all = load_excel(y_all)

    for subject in range(22):
        X_eye1 = []
        subject += 1
        # if subject==23:
        #   break
        #labeled——index 是场景序号1~5，labeled是场景名字
        for labeled_index, labeled in enumerate(labeled_list):
            # Delete missing data
            if subject == 8 and labeled_index == 3:
                continue
            if subject == 10 and labeled_index == 4:
                continue
            if subject == 11 and labeled_index == 2:
                continue
            if (subject == 19 and labeled_index == 4) or (subject == 21 and labeled_index == 0) or (
                    subject == 22 and labeled_index == 0):  # 数据不对应
                continue
            # load data and Select data
            data_cont = scio.loadmat(os.getcwd() + path_cont + 'subject' + str(subject) + '/' + str(labeled) + ".mat")
            data_eye = scio.loadmat(os.getcwd() + path_eye + 'Subject' + str(subject) + '/' + str(labeled) + ".mat")
            data_phy = scio.loadmat(os.getcwd() + path_phy + 'subject' + str(subject) + '/' + str(labeled) + ".mat")
            x_cont = data_cont['cz1'][5:, :]
            x_eye = data_eye['M2']
            x_phy = data_phy['z1'][5:, :]
            x_cont = np.delete(x_cont, [1, 3, 5, 6, 7, 8], axis=1)  # 7
            x_eye = np.delete(x_eye, [0, 2, 3, 6, 7, 8, 9], axis=1)  # 6
            x_phy = np.delete(x_phy, [0, 1, 2, 6, 7, 8], axis=1)  # 4
            # data_plot(x_cont, x_eye, x_phy)

            # Calculate the ratio of the amount of eye data to the amount of environmental data
            targets = x_cont[:, -1]
            ratio = x_eye.shape[0] / x_cont.shape[0]
            print(ratio, subject, labeled_index)

            # Remove abnormal data
            pupil_diams = []
            gaze_quals = []
            for index_cont in range(x_cont.shape[0]):
                if x_cont[index_cont, 1] > 5:
                    x_cont[index_cont, 1] = 3.5
                if x_cont[index_cont, 1] < 2:
                    x_cont[index_cont, 1] = 3.5
            for index_eye in range(x_eye.shape[0]):
                if x_eye[index_eye, 3] > 0.01:
                    x_eye[index_eye, 3] = 0.0087278
                if x_eye[index_eye, 0] > 20:
                    x_eye[index_eye, 0] = 0.49591
                if x_eye[index_eye, 0] > 2.5:
                    x_eye[index_eye, 0] = 0.74078

                # Average pupil diameter of left and right eyes
                pupil_diam = (x_eye[index_eye, 3] + x_eye[index_eye, 4]) / 2
                pupil_diams.append(pupil_diam)
                # Average gaze
                # gaze_qual = (x_eye[index_eye,0]+x_eye[index_eye,3])/2
                # gaze_quals.append(gaze_qual)
            x_eye = np.delete(x_eye, [3, 4], axis=1)  # 4
            x_eye = np.insert(x_eye, 3, values=pupil_diams, axis=1)  # 5
            # x_eye = np.insert(x_eye, 0, values=pupil_diams, axis=1)#8
            # data_plot(x_cont, x_eye, x_phy)

            # When the eye data is missing, the eye data is used as the benchmark to synchronize the data
            if ratio < 3:
                data_len = int((x_eye.shape[0] - (time * 3)) / (step * 3) + 1)
                # Set a sliding window to read data and labels
                for index in range(data_len):
                    save = 1
                    input_target = targets[int(index * step):int(index * step + time)]
                    # Find label boundaries
                    for i, target in enumerate(input_target):
                        if input_target[i] != input_target[i - 1]:
                            save = 0
                    # When a window data label is the same, save the data
                    if save == 1:
                        input_cont = x_cont[int(index * step):int(index * step + time), :]
                        input_eye = x_eye[int(index * step * 3):int(index * step * 3 + time * 3), :]
                        input_phy = x_phy[int(index * step):int(index * step + time), :]
                        if y_all[labeled_index][subject - 1] != 0:
                            y_step = y_all[labeled_index][subject - 1]
                            emotion_intensity.append(y_step)
                        X_cont.append(input_cont)
                        X_eye.append(input_eye)
                        X_phy.append(input_phy)
                        #y.append(y_step)
                        #load_excel()
            # When the environment data is missing, take the environment data as the benchmark and synchronize the data
            else:
                data_len = int((x_cont.shape[0] - time) / step + 1)
                # Set a sliding window to read data and labels
                for index in range(data_len):
                    save = 1
                    input_target = targets[int(index * step):int(index * step + time)]
                    # Find label boundaries
                    for i, target in enumerate(input_target):
                        if input_target[i] != input_target[i - 1]:
                            save = 0
                    # When a window data label is the same, save the data
                    if save == 1:
                        input_cont = x_cont[int(index * step):int(index * step + time), :]
                        input_eye = x_eye[int(index * step * 3):int(index * step * 3 + time * 3), :]
                        input_phy = x_phy[int(index * step):int(index * step + time), :]
                        #y_step = targets[int(index * step):int(index * step + 1)]
                        if y_all[labeled_index][subject-1]  != 0:
                            y_step = y_all[labeled_index][subject-1]
                            emotion_intensity.append(y_step)

                        X_cont.append(input_cont)
                        X_eye.append(input_eye)
                        X_phy.append(input_phy)
                        #load_excel()
                        #y.append(y_step)

        # X_eye1 = np.reshape(X_eye1,(-1,4))
        # scaler = MinMaxScaler(feature_range=(0, 1))#归一化
        # x_eye_subject = scaler.fit_transform(X_eye1[:,:-1])
        # X_eye.extend(x_eye_subject)
    #将数组转为二维数组
    emotion_intensity = np.array(emotion_intensity)
    emotion_intensity = emotion_intensity.reshape(len(emotion_intensity),1)
    X_cont = np.array(X_cont)
    X_eye = np.array(X_eye)
    X_phy = np.array(X_phy)
    #y = np.array(y)


    # Normalize the data
    X_cont = np.reshape(X_cont, (-1, 7))
    X_eye = np.reshape(X_eye, (-1, 5))
    X_phy = np.reshape(X_phy, (-1, 4))
    # data_plot(X_cont, X_eye, X_phy)
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))#归一化
    X_cont = scaler.fit_transform(X_cont[:,:-1]) 
    X_eye = scaler.fit_transform(X_eye[:,:-1]) 
    X_phy = scaler.fit_transform(X_phy[:,:-1]) 

    '''
    scaler = preprocessing.StandardScaler().fit(X_cont[:, :-1])
    X_cont = scaler.transform(X_cont[:, :-1])
    scaler = preprocessing.StandardScaler().fit(X_eye[:, :-1])
    X_eye = scaler.transform(X_eye[:, :-1])
    scaler = preprocessing.StandardScaler().fit(X_phy[:, :-1])
    X_phy = scaler.transform(X_phy[:, :-1])
    # data_plot(X_cont, X_eye, X_phy)

    # Store the sliding window data
    X_c = []
    X_e = []
    X_p = []
    for i in range(len(emotion_intensity)):
        input_cont = X_cont[i * time:i * time + time, :]
        input_eye = X_eye[i * time * 3:i * time * 3 + time * 3, :]
        input_phy = X_phy[i * time:i * time + time, :]
        X_c.append(input_cont)
        X_e.append(input_eye)
        X_p.append(input_phy)
    X_c = np.array(X_c)
    X_e = np.array(X_e)
    X_p = np.array(X_p)

    scio.savemat(os.getcwd() + inputs_path + 'cont_5s' + ".mat", {'cont': X_c})
    scio.savemat(os.getcwd() + inputs_path + 'eye_5s' + ".mat", {'eye': X_e})
    scio.savemat(os.getcwd() + inputs_path + 'phy_5s' + ".mat", {'phy': X_p})
    #y = y.transpose()
    scio.savemat(os.getcwd() + inputs_path + 'emotion_intensity_5s' + ".mat", {'emotion_intensity': emotion_intensity})


def data_plot(X_cont, X_eye, X_phy):
    groups = range(7)
    i = 1
    # plot each column
    # pyplot.figure(figsize=(20,10),frameon = False)
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(X_cont[:, group])
        # pyplot.title(group, y=1, loc='right')
        i += 1
        if group < 6:
            pyplot.xticks(())
            # pyplot.yticks(())
    pyplot.show()

    groups = range(5)
    i = 1
    # plot each column0
    # pyplot.figure(figsize=(20,10),frameon = False)
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(X_eye[:, group])
        # pyplot.title(group, y=1, loc='right')
        i += 1
        if group < 4:
            pyplot.xticks(())
            # pyplot.yticks(())
    pyplot.show()

    groups = range(4)
    i = 1
    # plot each column
    # pyplot.figure(figsize=(20,10),frameon = False)
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(X_phy[:, group])
        # pyplot.title(group, y=1, loc='right')
        i += 1
        if group < 3:
            pyplot.xticks(())
            # pyplot.yticks(())
    pyplot.show()

#加载情绪数据
def load_excel(y_all):
    wb = openpyxl.load_workbook(path_emotion_intensity)
    data = []  # 用于存放excel中所有的数据
    for rows in wb['Sheet3']:  # 对每个表单内的每一行遍历
        tmp = []  # 存放每一行的数据
        for cell in rows:  # 对每一行的每一个单元格进行遍历
            tmp.append(cell.value)  # 对一行数据进行存储
        data.append(tmp)  # 添加一整行数据
    for i in range(2, len(data)):
        index = 0
        for j in range(len(data[i])):
            #for index in range(5):

            if (data[1][j] == 'emotion intensity'):
                if (data[0][j] == 'scenario%d',index+1):

                    y_all[index].append(data[i][j])
                    index += 1
    return y_all


if __name__ == "__main__":
    processdata()

