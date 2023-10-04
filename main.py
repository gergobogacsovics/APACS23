import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse
import os
import time
import cv2
import math
from shutil import copyfile
from helper import ImageLoader, ImageDataset
from utils import LogLevel, Logger, Networks, get_network, ConfigLoader
from constants import COLOR_CODES_BY_CLASS
import signal
import sys
import json
import logging

config = ConfigLoader.load("config.yaml")

model_name = config["base"]["model"]
batch_size = config["base"]["hyper_params"]["batch_size"]
num_classes = config["base"]["hyper_params"]["classes"]
pixels_cut = config["base"]["hyper_params"]["pixels_cut"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


mpl.style.use("seaborn")

IMG_SAVE_DIR = "img_exports"

logging.info(f"Images will be saved to the following directory: {IMG_SAVE_DIR}.")

os.makedirs(IMG_SAVE_DIR)

model = get_network(model_name, num_classes, pixels_cut)

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")

logging.info(f"Using {torch.cuda.device_count()} GPUS.")

model = nn.DataParallel(model)
model.to(device)

if config["base"]["mode"] == "training":
    logging.info("Entering training mode.", LogLevel.INFO)
    
    train_dir_in = config["datasets"]["training"]["dir_inputs"]
    train_dir_out = config["datasets"]["training"]["dir_masks"]
    val_dir_in = config["datasets"]["validation"]["dir_inputs"]
    val_dir_out = config["datasets"]["validation"]["dir_masks"]

    logging.info(f"Using training directory: {train_dir_in}.")
    
    lr = config["modes"]["training"]["hyper_params"]["lr"]
    num_epochs = config["modes"]["training"]["hyper_params"]["epochs"]
    save_frequency = config["modes"]["training"]["checkpoints"]["saving_frequency"]
    saving_directory_networks = config["modes"]["training"]["checkpoints"]["saving_directory"] + "/" + model_name
    
    if not os.path.exists(saving_directory_networks):
        logging.info(f"Saving directory '{saving_directory_networks}' created.")
        os.makedirs(saving_directory_networks, exist_ok=True)
    
    train_dataset = ImageDataset(root_directory_input=train_dir_in, root_directory_output=train_dir_out, image_names_input=ImageLoader.load_image_names(directory=train_dir_in, extension=".jpg"), image_names_output=ImageLoader.load_image_names(directory=train_dir_out, extension=".png"))
    validation_dataset = ImageDataset(root_directory_input=val_dir_in, root_directory_output=val_dir_out, image_names_input=ImageLoader.load_image_names(directory=val_dir_in, extension=".jpg"), image_names_output=ImageLoader.load_image_names(directory=val_dir_out, extension=".png"))
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                               batch_size=batch_size, 
                                               shuffle=False)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    total_steps_training = len(train_loader)
    total_steps_val = len(validation_loader)
    
    losses = np.zeros(num_epochs)
    validation_losses = np.zeros(num_epochs)

    min_val_loss = math.inf
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        progress_bar_train = tqdm(total=len(train_loader))
        
        model.train()
        
        for _, images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
    
            outputs = model(images)
    
            loss = criterion(outputs, labels)
    
            total_loss += loss.item()
    
            loss.backward()
            optimizer.step()
            
            progress_bar_train.update(1)
        
        progress_bar_train.close()
        
        print ('Epoch [{}/{}], Training Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / total_steps_training))
        
        losses[epoch] = total_loss / total_steps_training
        
        # Validation part
        total_loss_val = 0
        
        progress_bar_val = tqdm(total=len(validation_loader))
        
        model.eval()
        
        with torch.no_grad():
            for _, images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)
        
                outputs =  model(images)
        
                loss = criterion(outputs, labels)
        
                total_loss_val += loss.item()
                
                progress_bar_val.update(1)
        
        progress_bar_val.close()
        
        print ('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss_val / total_steps_val))
        
        validation_losses[epoch] = total_loss_val / total_steps_val
    
        if total_loss_val < min_val_loss:
            logging.info("Saving new best model.")
            
            torch.save(model.state_dict(), f"{saving_directory_networks}/network_val_{int(time.time())}__loss_{total_loss}__val_loss_{total_loss_val}.pth")
            
            min_val_loss = total_loss_val

        if save_frequency != -1 and ((epoch + 1) % save_frequency == 0):
            logging.info("Saving model.")
            
            torch.save(model.state_dict(), f"{saving_directory_networks}/network_{int(time.time())}.pth")

    logging.info("Training DONE.")

    logging.info("Saving model.")
            
    torch.save(model.state_dict(), f"{saving_directory_networks}/network_final_{int(time.time())}.pth")

    logging.info("Plotting results.")

    epochs = range(1, num_epochs + 1)

    plt.title("Losses")
    plt.plot(epochs, losses, "c-")
    plt.plot(epochs, validation_losses, "-", color="orange")
    plt.legend(["Loss", "Val Loss"])
    plt.show()
else:
    logging.info("Entering test mode.")
    
    test_dir_in = config["datasets"]["test"]["dir_inputs"]
    test_dir_out = config["datasets"]["test"]["dir_masks"]
    model_checkpoint_path = config["modes"]["test"]["checkpoint"]
    tag = config["modes"]["test"]["tag"]
    saving_directory_test = config["modes"]["test"]["saving_directory"] + "/" + model_name + "_" + tag
    
    logging.info(f"Loading model '{model_checkpoint_path}'")
    model.load_state_dict(torch.load(model_checkpoint_path))

    if not os.path.exists(saving_directory_test):
        logging.info(f"Saving directory '{saving_directory_test}' created.")
        os.makedirs(saving_directory_test, exist_ok=True)
    
    test_dataset = ImageDataset(root_directory_input=test_dir_in, root_directory_output=test_dir_out, image_names_input=ImageLoader.load_image_names(directory=test_dir_in, extension=".jpg"), image_names_output=ImageLoader.load_image_names(directory=test_dir_out, extension=".png"))

    # Data loaders
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size, 
                                               shuffle=True)
    
    logging.info("Running test.")

    progress_bar_val = tqdm(total=len(test_loader))
        
    model.eval()
        
    with torch.no_grad():
        for image_names, images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs =  model(images).data.cpu().numpy()

            for img_name, image, label, output in zip(image_names, images, labels, outputs):
                output = output[0]
                
                output_image = np.stack(((output >= 0) * 255,)*3, axis=-1)

                img_file_name = img_name.split("/")[-1]
                file_name = f"{saving_directory_test}/{img_file_name}.jpg"
  
                cv2.imwrite(file_name, output_image)
                
            progress_bar_val.update(1)
        
    progress_bar_val.close()

    logging.info("Test DONE.")
