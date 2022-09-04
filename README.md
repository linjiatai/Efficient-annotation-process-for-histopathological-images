# WSI-loop

## Introduction
This program is a high-efficiency annotation process. The junior pathologist can self-accumulate a large number of pixel-level annotations by this process. and shubmit them to the senior pathologist for final modification and confirmation as ground-truth for deep learning.

## Preparation

- You should download the pre-trained weight from XXX and put it in fold XXX.
- And you should prepare your region of interests (ROIs) and put them in the fold XXX.
- Third, you should confirme the tissue types of your data and modify the palette in this program, such as:
palette = [0]*100
palette[0:3] = [255,255,255]
palette[3:6] = [120,120,120]
palette[6:9] = [255,0,0]
palette[9:12] = [0,255,0]
palette[12:15] = [0,255,255]
palette[15:18] = [255,0,255]
palette[18:21] = [237,145,33]

## Usage

