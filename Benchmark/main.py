
"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - see AUTHORS file

This file is part of the Online Distillation software.

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

import os
import torch
from tqdm import tqdm

import torch.optim as optim

from evaluation import SegmentationEvaluator, write_segmentation_performance

import sys

def write_log_args(args, save_path):
    f = open(os.path.join(save_path, 'arguments'), 'w')
    for key, value in vars(args).items():
        f.write(f'{key} = {value}\n')
    f.close()
    

if __name__ == "__main__":

        # --------------------
        # Import the arguments
        # --------------------

        from arguments import args

        # ------------------------
        # Create the saving folder
        # ------------------------

        saving_counter = 1
        save_path = args.save
        if args.subfolder is not None:
            os.makedirs(args.save + "/" + args.subfolder, exist_ok=True)
            save_path = args.save + "/" + args.subfolder
        while os.path.isdir(save_path + "/experiment_" + str(saving_counter)):
            saving_counter += 1
        save_path = save_path + "/experiment_" + str(saving_counter)
        os.mkdir(save_path)
        
        # -------------------------
        # Create output folder of the student masks
        # -------------------------
        
        os.makedirs(os.path.join(args.dataset, 'pseudo_groundtruth_seg', args.teacher, 'mask', args.onlinedataset + '-' + args.trainer), exist_ok=True)
        
        # -------------------------
        # Write arguments in a file
        # -------------------------
        
        write_log_args(args, save_path)
        
        # -----------------------
        # Defining some variables
        # -----------------------

        device = torch.device(args.device)
        num_classes = args.numclasses

        # --------------------------------------
        # Initializing the network and optimizer
        # --------------------------------------

        # Building the network
        from networks.tinynet import TinyNet
        network = TinyNet(num_classes=num_classes).to(device)
        
        total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        parameters_per_layer  = [p.numel() for p in network.parameters() if p.requires_grad]
        print("Total number of parameters: " + str(total_params))

        # Definition of the optimizer
        optimizer = optim.Adam(network.parameters(), lr=args.learningrate)

        # Get the weights for the class imbalance and update the criterion
        #weights = compute_weights(labels).to(device)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # Instantiate the network trainer
        
        network_trainer = None
        
        if args.trainer == "MAS":
            from training.mas import MASNetworkTrainer
            network_trainer = MASNetworkTrainer(network, optimizer, criterion, device, args)
        elif args.trainer == "ACE":
            from training.ace import ACENetworkTrainer
            network_trainer = ACENetworkTrainer(network, optimizer, criterion, device, args)
        elif args.trainer == "LWF":
            from training.lwf import LWFNetworkTrainer
            network_trainer = LWFNetworkTrainer(network, optimizer, criterion, device, args)
        elif args.trainer == "RWalk":
            from training.rwalk import RWalkNetworkTrainer
            network_trainer = RWalkNetworkTrainer(network, optimizer, criterion, device, args)
        else:
            from training.classic import ClassicNetworkTrainer
            network_trainer = ClassicNetworkTrainer(network, optimizer, criterion, device, args)
        
        # log file
        evaluator = SegmentationEvaluator(n_classes = num_classes)
        write_segmentation_performance(os.path.join(save_path, "performance.log"), evaluator, None, None, init=True)
        
        
        # ------------------------
        # Initializing the dataset
        # ------------------------

        dataset = None
        
        if args.onlinedataset == "fifo":
            from datasets.fifo import FifoDataset
            dataset = FifoDataset(args)
        elif args.onlinedataset == "uniform":
            from datasets.uniform import UniformDataset
            dataset = UniformDataset(args)
        elif args.onlinedataset == "prioritized":
            from datasets.prioritized import PrioritizedDataset
            dataset = PrioritizedDataset(args, network, criterion)
        elif args.onlinedataset == "er":
            from datasets.er import ExperienceReplay
            dataset = ExperienceReplay(args)
        elif args.onlinedataset == "per":
            from datasets.per import PrioritizedExperienceReplay
            dataset = PrioritizedExperienceReplay(args, network, criterion)
        elif args.onlinedataset == "mir":
            from datasets.mir import MIR
            dataset = MIR(args, network, criterion, optimizer)
        elif args.onlinedataset == "efifo":
            from datasets.efifo import EFifoDataset
            dataset = EFifoDataset(args)
        else:
            raise Exception("No valid online dataset has been specified.")
        
        
        pbar = tqdm(total=len(dataset), ncols=120, desc="OD")
        # Train the network and evaluate the different updates
        while dataset.current_position < len(dataset):
            
            # -------------
            # Training part
            # -------------

            dataloader = dataset.create_trainset()
            network_trainer.train_epoch(dataloader)

            # -------------
            # Testing part
            # -------------
            
            # Evaluate frames in the interval [t, t+studentsubsample]
            dataloader = dataset.create_testset()
            evaluator_now = SegmentationEvaluator(n_classes = num_classes)
            
            network_trainer.evaluate(dataloader, evaluator_now)
            
            # Evaluate frames in the interval [t-sequence_length, t-sequence_length+studentsubsample]
            start = dataset.current_position-args.sequence_length*args.studentsubsample
            end = start + args.testingsetsize
            dataloader = dataset.create_custom_set(start, end)
            evaluator_before = SegmentationEvaluator(n_classes = num_classes)
            
            network_trainer.evaluate(dataloader, evaluator_before)
            
            # Evaluate frames in the interval [t+sequence_length, t+sequence_length+studentsubsample]
            start = dataset.current_position+args.sequence_length*args.studentsubsample
            end = start + args.testingsetsize
            dataloader = dataset.create_custom_set(start, end)
            evaluator_after = SegmentationEvaluator(n_classes = num_classes)
            
            network_trainer.evaluate(dataloader, evaluator_after)
            
            # Save results
            
            write_segmentation_performance(os.path.join(save_path, "performance.log"), evaluator_now, evaluator_before, evaluator_after)

            torch.cuda.empty_cache()

            # Perform a step in the online data set
            
            dataset.step()
            pbar.update(args.studentsubsample)
        
        
        
        # Evaluate the final ARTHuS on the whole video sequence
        from datasets.fifo import FifoDataset
        dataset = FifoDataset(args)
        
        # log file
        evaluator = SegmentationEvaluator(n_classes = num_classes)
        write_segmentation_performance(os.path.join(save_path, "final.log"), evaluator, None, None, init=True)
        
        pbar = tqdm(total=len(dataset), ncols=120, desc="Final evaluation")
        while(dataset.current_position < len(dataset)):
            
            # Init segmentation evaluator
            evaluator = SegmentationEvaluator(n_classes = num_classes)
            
            # Evaluate frames in the interval [t, t+studentsubsample]
            dataloader = dataset.create_testset()
            network_trainer.evaluate(dataloader, evaluator)
            
            # Save results
            write_segmentation_performance(os.path.join(save_path, "final.log"), evaluator, evaluator, evaluator)

            torch.cuda.empty_cache()
            
            # Perform a step in the online data set
            dataset.step()
            pbar.update(args.studentsubsample)
            
        