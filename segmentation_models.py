
# Define a class to represent different types of segmentation models
class SegmentationModels:
    """
    A class to encapsulate the names of various segmentation models.
    Each attribute of this class represents a type of segmentation model.
    
    This class acts as a simple enumeration for different models, providing
    a convenient way to refer to them by a single name.
    """
    
    Unet = 'Unet'
    UnetPlusPlus = 'UnetPlusPlus'
    MAnet = 'MAnet'
    Linknet = 'Linknet'
    FPN = 'FPN'
    PSPNet = 'PSPNet'
    DeepLabV3 = 'DeepLabV3'
    DeepLabV3Plus = 'DeepLabV3Plus'
    PAN = 'PAN'