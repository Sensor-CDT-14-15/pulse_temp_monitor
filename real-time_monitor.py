# -*- coding: utf-8 -*-
"""
Plots live temperature data and pulse from Arduino
"""
import serial
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.signal import convolve

arduinoData = serial.Serial('COM3', 9600)

# Length of arrays and graphs
maxlen = 200

# Measurements per second
Fs = 40

# Arrays for plots
xtemp =    []
xRpulse =  []
xIRpulse = []
ytemp =    []
yRpulse =  []
yIRpulse = []
# Arrays for averaged data
temps =    []
tempsav =  []
Rpulses =  []
Rpulsav =  []
IRpulses = []
IRpulsav = []
yFFT =    [0]

fig = plt.figure()

tempav = 0
Rpulsav = 0
IRpulsav = 0

# Temp
temp_axes = fig.add_subplot(2,2,1)
temp_axes.set_xlim(0,maxlen)
temp_axes.set_ylim(20,40)
temp_axes.set_title("Temperature in degrees Celsius")
temp_axes.text(1870,30,("temp"),fontsize=10,color="r")
temp_axes.grid()
temp_line, = temp_axes.plot([], [], color='b')

# Pulse
pulse_axes = fig.add_subplot(2,2,2)
pulse_axes.set_xlim(0,maxlen)
pulse_axes.set_ylim(0,800)
pulse_axes.set_title("Raw diode data")
pulse_axes.grid()
pulse_line_1, = pulse_axes.plot([], [], color='r')
pulse_line_2, = pulse_axes.plot([], [], color='g')

# FFT
fft_axes = fig.add_subplot(2,2,3)
fft_axes.set_xlim(0,5)
fft_axes.set_ylim(0,300)
fft_axes.set_title("Live FFT of diode data")
fft_line_1, = fft_axes.plot([], [])
fft_line_2, = fft_axes.plot([], [])


# Digit display
digit_axes = fig.add_subplot(2,2,4)
digit_axes.set_xlim(-1,9)
digit_axes.set_ylim(-7,5)
digit_axes.set_title("Current pulse and temperature")
digit_line_1, = digit_axes.plot([], [])
digit_line_2, = digit_axes.plot([], [], color='r')

coords = [
[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)],
[(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
[(2, 0), (1, 0), (0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (1, 4), (0, 4)],
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4), (1, 4), (0, 4)],
[(0, 4), (0, 3), (0, 2), (1, 2), (2, 2), (2, 3), (2, 2), (2, 1), (2, 0)],
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4)],
[(2, 4), (1, 4), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)],
[(0, 4), (1, 4), (2, 4), (2, 3), (1, 2), (1, 1), (1, 0)],
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2)],
[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (2, 3), (2, 2)]
]
def number_coords(num):
    num_str = str(num)
    if (len(num_str) == 1):
        num_coords = ([x[0] + 6 for x in coords[num]], [x[1] for x in coords[num]])
    elif(len(num_str) == 2):
        x_coords = [x[0] + 3 for x in coords[int(num_str[0])]]
        y_coords = [x[1] for x in coords[int(num_str[0])]]
        x_coords_new = [x[0] + 6 for x in coords[int(num_str[1])]]
        y_coords_new = [x[1] for x in coords[int(num_str[1])]]
        x_coords = x_coords + [np.nan] + x_coords_new
        y_coords = y_coords + [np.nan] + y_coords_new
        num_coords = [x_coords, y_coords]
    elif(len(num_str) == 3):
        x_coords = [x[0] for x in coords[int(num_str[0])]]
        y_coords = [x[1] for x in coords[int(num_str[0])]]
        x_coords_new = [x[0] + 3 for x in coords[int(num_str[1])]]
        y_coords_new = [x[1] for x in coords[int(num_str[1])]]
        x_coords = x_coords + [np.nan] + x_coords_new
        y_coords = y_coords + [np.nan] + y_coords_new
        x_coords_new = [x[0] + 6 for x in coords[int(num_str[2])]]
        y_coords_new = [x[1] for x in coords[int(num_str[2])]]
        x_coords = x_coords + [np.nan] + x_coords_new
        y_coords = y_coords + [np.nan] + y_coords_new
        num_coords = [x_coords, y_coords]
    else:
        num_coords = [np.nan, np.nan]
    return num_coords


def FourierTransformArray(Rpulses):
    fft_y = np.fft.fft(Rpulses)
    n = len(fft_y)
    freq = np.fft.fftfreq(n, 1/40.0)
    
    half_n = np.ceil(n/2.0)
    fft_y_half = (2.0 / n) * fft_y[:half_n]
    freq_half = freq[:half_n]
    
    return freq_half, np.abs(fft_y_half),


def find_peaks(data):
    # Ensure that we have enough data to convolve with our kernel
    assert (len(data) > 4), "At least five data points must be passed to find_peaks as the kernel has three elements."
    kernel = [1, 0, -1]
    dData = convolve(data, kernel, 'valid')
    S = np.sign(dData)
    ddS = convolve(S, kernel, 'valid')
    # These candidates are basically all negative slope positions
    # Add one since using 'valid' shrinks the arrays
    candidates = np.where(dData < 0)[0] + (len(kernel) - 1)
    # Here they are filtered on actually being the final such position in a run of negative slopes
    indices = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 1))
    return indices


def init():
    temp_line.set_data([], [])
    pulse_line_1.set_data([], [])
    pulse_line_2.set_data([], [])
    fft_line_1.set_data([], [])
    fft_line_2.set_data([], [])
    digit_line_1.set_data([], [])
    digit_line_2.set_data([], [])
    return temp_line, pulse_line_1, pulse_line_2, fft_line_1, fft_line_2, digit_line_1, digit_line_2


def animate(i):
    global pulses
    global freq
    global tempav
    global Rpulsav
    global IRpulsav
    
    freq = 0
    
    #Get data from arduino, separate pulse and temp into arrays
    arduinoString = arduinoData.readline()
    dataArray = arduinoString.split("\t")
    
    Rpulse = float(dataArray[0])
    temp =  float(dataArray[1])
    IRpulse = float(dataArray[2])
    
    # Get an averaged temp and pulse
    if (len(temps) <= 3) & (len(Rpulses) & (len(IRpulses)) <= 2):
        temps.append(temp)
        Rpulses.append(Rpulse)
        IRpulses.append(IRpulse)
    else:
        Rpulses.append(Rpulse)
        IRpulses.append(IRpulse)
        temps.append(temp)
        Rpulses.pop(0)
        IRpulses.pop(0)
        temps.pop(0)
        Rpulsav = sum(Rpulses)/len(Rpulses)
        IRpulsav = sum(IRpulses)/len(IRpulses)
        tempav = sum(temps)/len(temps)
    
    # Keeps temp and pulse data set at 2000 points max
    if (len(ytemp)<=maxlen) & (len(yRpulse)<=maxlen) & (len(yIRpulse)<=maxlen):
        xtemp.append(i)
        xRpulse.append(i)
        xIRpulse.append(i)
        ytemp.append(tempav)
        yRpulse.append(Rpulsav)
        yIRpulse.append(IRpulsav)
        yFFT.append(Rpulsav)
    else:
        ytemp.pop(0)
        ytemp.append(tempav)
        yRpulse.pop(0)
        yRpulse.append(Rpulsav)
        yIRpulse.pop(0)
        yIRpulse.append(IRpulsav)
        yFFT.pop(0)
        yFFT.append(Rpulsav)
    
    freqs_red = FourierTransformArray(yRpulse)[0]
    amplitudes_red = FourierTransformArray(yRpulse)[1]
    freqs_ir = FourierTransformArray(yIRpulse)[0]
    amplitudes_ir = FourierTransformArray(yIRpulse)[1]
    
    #Send averaged data to be stored lines to be plotted    
    temp_line.set_data(xtemp, ytemp)
    pulse_line_1.set_data(xRpulse, yRpulse)
    pulse_line_2.set_data(xIRpulse, yIRpulse)
    fft_line_1.set_data(freqs_red, amplitudes_red)
    fft_line_2.set_data(freqs_ir, amplitudes_ir)
    
    if(len(amplitudes_red) > 4):
        amplitude_peak_indices = find_peaks(amplitudes_red)
        pulse_freq_red = [i for i in freqs_red[amplitude_peak_indices] if i > 0.5]
        if(len(pulse_freq_red)):
            digit_line_1.set_data(number_coords(int(60 / pulse_freq_red[0])))
            
    if(len(amplitudes_ir) > 4):
        amplitude_peak_indices = find_peaks(amplitudes_ir)
    
    temp_coords = number_coords(int(temp * 10))
    temp_coords[1] = [x - 6 for x in temp_coords[1]]
    temp_coords[0] = temp_coords[0] + [np.nan, 5.45, 5.55, 5.55, 5.45, 5.45]
    temp_coords[1] = temp_coords[1] + [np.nan, -3.95, -3.95, -4.05, -4.05, -3.95]
    digit_line_2.set_data(temp_coords)
    
    return temp_line, pulse_line_1, pulse_line_2, fft_line_1, fft_line_2, digit_line_1, digit_line_2

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=3000, interval=20, blit=True)

fig.tight_layout()
plt.show()
