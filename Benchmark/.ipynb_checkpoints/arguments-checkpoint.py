"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - see AUTHORS file

This file is part of the GPS Online Distillation software.

This program is free software: you can redistribute it and/or modify it under the terms 
of the GNU Affero General Public License as published by the Free Software Foundation, 
either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this 
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Path to the dataset folder", required=True)
parser.add_argument("--numclasses", help="Number of classes", type=int)
parser.add_argument("--save", help="Path to the folder for saving the results")
parser.add_argument("--subfolder", help="Subfolder for saving the results", default=None)
parser.add_argument("--teacher", help="teacher folder")
parser.add_argument("--groundtruth", help="groundtruth folder")
parser.add_argument("--start", help="Only select a subset of the dataset (starting point)", type = int, default=None)
parser.add_argument("--stop", help="Only select a subset of the dataset (end point)", type = int, default=None)


parser.add_argument("--datasetsize", help="Size of the online dataset", type = int, default=67)
parser.add_argument("--numworkers", help="number of workers for the dataloader", type = int, default=4)
parser.add_argument("--batchsize", help="Batch size", type = int, default=8)
parser.add_argument("--testingsetsize", help="Size of the test dataset", type = int, default=20)
parser.add_argument("--learningrate", help="Learning rate", type = float, default=0.0001)
parser.add_argument("--device", help="Device to use (use cuda:x or cpu)", default="cuda:0")
parser.add_argument("--teachersubsample", help="Subsampling for the teacher, number of frames to skip", type = int, default=75)
parser.add_argument("--studentsubsample", help="Subsampling for the student training, number of teacher frames to skip", type = int, default=20)

# Type of the online data set
parser.add_argument("--onlinedataset", help="Type of online dataset to use", required=True)

# Type of training
parser.add_argument("--trainer", help="Trainer type", default='')

# Online adaptation for regularization-based methods (MAS, RWalk and LWF)
parser.add_argument("--update-freq", help="(MAS, RWalk and LWF): Rate (in epoch) at which an update must be triggered", type=int, default=1)
parser.add_argument("--warmup", help="(MAS, RWalk and LWF): Number of epochs to wait before triggering the first update", type=int, default=10)

parser.add_argument("--er-capacity", help="(Replay-based): Capacity of the replay buffer", type=int, default=250)

parser.add_argument("--alpha", type=float, 
                    help='(Prioritized): Adjust the conservation of frames \n (LWF): scaling factor on the regularizer \n (RWalk): Exponential averaging factor for the importance weights',
                    default=0.5)

parser.add_argument("-C", type=int, help="(MIR): Subsample of the replay buffer", default=150)
parser.add_argument("-k", type=int, help="(MIR): Number of samples to evaluate", default=80)

parser.add_argument("--reg", help='(MAS): Factor determining the importance of the weight loss', type=float, default=1)
parser.add_argument("--decay", help='(MAS): Exponential averaging factor for the importance weights', type=float, default=0.1)

# LWF arguments
parser.add_argument("--temperature", help="(LWF): Adjust the temperature of the softmax layer", type=float, default=2.)

# RWalk arguments
parser.add_argument("--delta-t", help="(RWalk): Rate (in iterations) at which a score update must be triggered", type=int)

# time shift to compute BWT and FWT over time
parser.add_argument("--sequence-length", help="Length of a sequence (in minutes)", type=int, default=20)

args = parser.parse_args()
