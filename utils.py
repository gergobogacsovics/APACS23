from enum import Enum
import yaml
from cerberus import Validator
from nets import FCN8, FCN16, FCN32, CombinedNet6Channels, CombinedNet5Channels3216, CombinedNet5Channels168, CombinedNet5Channels328
import sys
import logging

class Networks(Enum):
    FCN_8 = "FCN-8"
    FCN_16 = "FCN-16"
    FCN_32 = "FCN-32"
    COMBINED_6CH = "CombinedNetwork-6ch"
    CBOMINED_5CH3216 = "CombinedNetwork-5ch-32-16"
    CBOMINED_5CH328 = "CombinedNetwork-5ch-32-8"
    CBOMINED_5CH168 = "CombinedNetwork-5ch-16-8"
    
def get_network(name, num_classes, pixels_cut=None):
    if name == Networks.FCN_8.value:
        return FCN8(number_of_classes=num_classes, pixels_cut=pixels_cut)
    elif name == Networks.FCN_16.value:
        return FCN16(number_of_classes=num_classes, pixels_cut=pixels_cut)
    elif name == Networks.FCN_32.value:
        return FCN32(number_of_classes=num_classes, pixels_cut=pixels_cut)
    elif name == Networks.COMBINED_6CH.value:
        return CombinedNet6Channels(number_of_classes=num_classes, pixels_cut=pixels_cut,
        fcn_32_path="fill_me",
        fcn_16_path="fill_me",
        fcn_8_path="fill_me")
    elif name == Networks.CBOMINED_5CH168.value:
        return CombinedNet5Channels168(number_of_classes=num_classes, pixels_cut=pixels_cut,
        fcn_16_path="fill_me",
        fcn_8_path="fill_me")
    elif name == Networks.CBOMINED_5CH3216.value:
        return CombinedNet5Channels3216(number_of_classes=num_classes, pixels_cut=pixels_cut,
        fcn_32_path="fill_me",
        fcn_16_path="fill_me")
    elif name == Networks.CBOMINED_5CH328.value:
        return CombinedNet5Channels328(number_of_classes=num_classes, pixels_cut=pixels_cut,
        fcn_32_path="fill_me",
        fcn_8_path="fill_me")
    else:
        logging.error(f"Unknown network name '{name}'")
        
class ConfigLoader:
    @classmethod
    def load(cls, path, schema_path="schema.py"):
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
            
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f)
        
        validator = Validator(schema)
        
        is_valid = validator.validate(doc, schema)
        
        if not is_valid:
            logging.error(f"Could not validate '{path}'.")
            logging.error(f"The error was: {validator.errors}'.")
            
            sys.exit(-1)
        
        logging.info("Config file validated successfully.")
        
        return doc