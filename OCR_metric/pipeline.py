from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
from pytesseract import pytesseract
import re
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
import re
import nltk
from nltk.corpus import words
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error


try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

english_vocab = set(words.words())


tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Adjust this for your system
pytesseract.tesseract_cmd = tesseract_cmd
custom_config = r'--oem 1 --psm 3'  #psm 3,11,6

def is_valid_word(word, min_length=1):
    """Check if word is alphanumeric and long enough."""
    return bool(re.match(r'^[a-zA-Z0-9]+$', word)) and len(word) >= min_length and word.lower() in english_vocab 


def get_bounding_boxes(gt_img, min_width=5, min_height=5, vis=True, bb_name=''):
    '''Detect text regions in the image and return bounding boxes and detected texts.
    Parameters:
        gt_img: Input image.
        min_width: Minimum width of bounding boxes to consider.
        min_height: Minimum height of bounding boxes to consider.
        vis: Whether to visualize the bounding boxes on the image.
        bb_name: Name identifier for saving visualizations.
    Returns:
        bounding_boxes: List of bounding boxes (x, y, w, h).
        detected_texts: List of detected text strings corresponding to the bounding boxes.
    '''
    data = pytesseract.image_to_data(gt_img, output_type=pytesseract.Output.DICT,config=custom_config)
    bounding_boxes = []
    detected_texts = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        raw_conf = data['conf'][i]
        if raw_conf == '-1': 
            continue
        conf = int(raw_conf)
        if conf < 70:
            continue 
        text = data['text'][i].strip()
        if not text: 
            continue
        (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
        if len(text) > 0 and w >= min_width and h >= min_height and is_valid_word(text,min_length=1):
            bounding_boxes.append((x, y, w, h))
            detected_texts.append(text)

    if vis:
        gt_image_with_boxes = gt_img.copy()
        draw = ImageDraw.Draw(gt_image_with_boxes)
        for (x, y, w, h) in bounding_boxes:
            draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(gt_image_with_boxes)
        plt.title(f"Detected {len(bounding_boxes)} bounding boxes")
        plt.axis("off")
        plt.savefig(f"path/to/save/bounding_boxes/bbox_{bb_name}.png", bbox_inches="tight", dpi=300)
        plt.show()

        print("Sample detected bounding boxes (x, y, w, h):")
        print(bounding_boxes[:5])
        print("Sample detected texts:")
        print(detected_texts[:5])
        
    return bounding_boxes, detected_texts

def crop_bounding_boxes(gt_img, sr_img, bounding_boxes, vis=True, num_vis=3):
    '''Crop regions from GT and SR images based on bounding boxes.
    Parameters:
        gt_img: Ground truth image.
        sr_img: Super-resolved image.
        bounding_boxes: List of bounding boxes (x, y, w, h).
        vis: Whether to visualize some of the cropped regions.
        num_vis: Number of cropped regions to visualize.
    Returns:
        cropped_pairs: List of tuples containing cropped GT and SR images.
    '''
    cropped_pairs = []
    
    for (x, y, w, h) in bounding_boxes:
        gt_crop = gt_img.crop((x, y, x + w, y + h))
        sr_crop = sr_img.crop((x, y, x + w, y + h))
        cropped_pairs.append((gt_crop, sr_crop))
    
    if vis:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(num_vis, 2, figsize=(8, num_vis * 3))
        for idx in range(min(num_vis, len(cropped_pairs))):
            gt_crop, sr_crop = cropped_pairs[idx]
            axes[idx, 0].imshow(gt_crop, cmap='gray')
            axes[idx, 0].set_title(f"GT Crop {idx+1}")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(sr_crop, cmap='gray')
            axes[idx, 1].set_title(f"SR Crop {idx+1}")
            axes[idx, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return cropped_pairs

def ocr_2nd_stage(detected_texts,cropped_pairs, vis=True, config = custom_config, num_vis = 10):
    '''Perform OCR on cropped SR images and pair with GT texts.
    Parameters:
        detected_texts: List of texts detected in the GT image.
        cropped_pairs: List of tuples containing cropped GT and SR images.
        vis: Whether to visualize some OCR results.
        config: Tesseract OCR configuration string.
        num_vis: Number of OCR results to visualize.
    Returns:
        ocr_result: List of dictionaries with GT and SR texts.
    '''
    ocr_result = []
    for idx, (gt_crop, sr_crop) in enumerate(cropped_pairs):
        gt_text = detected_texts[idx] 
        if sr_crop.height> sr_crop.width:
            sr_crop = sr_crop.rotate(-90, expand=True)
        sr_text = pytesseract.image_to_string(sr_crop, config="--oem 1 --psm 7 -l eng").strip()
        ocr_result.append({"GT_text": gt_text, "SR_text":sr_text})
    if vis:
        print("Sample OCR Results from Crops:")
        for i in range(min(num_vis, len(ocr_result))):
            print(f"\n--- Crop {i+1} ---")
            print(f"GT Text: {ocr_result[i]['GT_text']}")
            print(f"SR Text: {ocr_result[i]['SR_text']}")

    return ocr_result



def eval_ocr_results(ocr_result, cropped_pairs, verbose=True):
    '''Evaluate OCR results using various metrics.
    Parameters:
        ocr_result: List of dictionaries with GT and SR texts.
        cropped_pairs: List of tuples containing cropped GT and SR images.
        verbose: Whether to print detailed results for the first few crops.
    Returns:
        metrics: Dictionary with mean values of various metrics.
    '''
    cer_scores = []
    wer_scores = []
    coverage_scores = []
    soft_coverage_scores = []
    classic_coverages = [] ###
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    
    for i,result in enumerate(ocr_result):
        gt = result['GT_text']
        sr = result['SR_text']
        gt_crop, sr_crop = cropped_pairs[i]
        
        cer = calc_CER(gt, sr)
        wer = calc_WER(gt, sr)
        classic_coverage = calc_classic_coverage(gt, sr)
        coverage = calc_text_coverage(gt, sr)
        soft_coverage = calc_soft_coverage(gt, sr)
        
        psnr_val, ssim_val, mse_val = compute_pixel_metrics(gt_crop, sr_crop)
        
        cer_scores.append(cer)
        wer_scores.append(wer)
        classic_coverages.append(classic_coverage)
        coverage_scores.append(coverage)
        soft_coverage_scores.append(soft_coverage)
        psnr_scores.append(psnr_val)
        ssim_scores.append(ssim_val)
        mse_scores.append(mse_val)
        
        if verbose and i < 3:  
            print(f"\n--- Crop {i+1} ---")
            print(f"GT: {gt}")
            print(f"SR: {sr}")
            print(f"CER: {cer:.3f}, WER: {wer:.3f}, Coverage: {coverage:.3f}, Soft Coverage: {soft_coverage:.3f}, Classic Coverage: {classic_coverage:.3f}")
            print(f"PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.3f}, MSE: {mse_val:.2f}")

    if not cer_scores or not wer_scores:          # no valid crops
        return {
            "Mean CER": 1.0,
            "Mean WER": 1.0,
            "Mean Coverage": 0.0,
            "Mean Soft Coverage": 0.0,
            "Mean PSNR": float("-inf"),
            "Mean SSIM": 0.0,
            "Mean MSE": float("inf"),
        }


    return {
        "Mean CER": sum(cer_scores) / len(cer_scores),
        "Mean WER": sum(wer_scores) / len(wer_scores),
        "Mean Classic Coverage": sum(classic_coverages) / len(classic_coverages),  ###
        "Mean Coverage": sum(coverage_scores) / len(coverage_scores),
        "Mean Soft Coverage": sum(soft_coverage_scores) / len(soft_coverage_scores),
        "Mean PSNR": np.mean(psnr_scores),
        "Mean SSIM": np.mean(ssim_scores),
        "Mean MSE": np.mean(mse_scores),
    }

def compute_pixel_metrics(gt_crop, sr_crop):
    '''Compute PSNR, SSIM, and MSE between GT and SR cropped images.
    Parameters:
        gt_crop: Cropped ground truth image.
        sr_crop: Cropped super-resolved image.
    Returns:
        psnr_val: Peak Signal-to-Noise Ratio.
        ssim_val: Structural Similarity Index.
        mse_val: Mean Squared Error.
    '''
    gt_np = np.array(gt_crop.convert("L"))
    sr_np = np.array(sr_crop.convert("L"))

    mse_val = mean_squared_error(gt_np.flatten(), sr_np.flatten())
    if mse_val == 0:
        return float('inf'), 1.0, mse_val  # Avoid division by zero in PSNR
    psnr_val = psnr(gt_np, sr_np, data_range=255)
    h, w = gt_np.shape
    max_win = min(h, w)
    win = max_win if (max_win % 2 == 1 and max_win >= 3) else (max_win - 1 if (max_win - 1) >= 3 else 3)
    ssim_val = ssim(gt_np, sr_np, data_range=255, win_size=win)

    return psnr_val, ssim_val, mse_val
    

def calc_score_v2(gt_path, sr_path, vis=False, bb_name=''):
    '''Complete OCR evaluation pipeline from image paths.
    Parameters:
        gt_path: Path to the ground truth image.
        sr_path: Path to the super-resolved image.
        vis: Whether to visualize intermediate steps.
        bb_name: Name identifier for saving bounding box visualizations.'''
    gt_img = Image.open(gt_path)
    sr_img = Image.open(sr_path)
    gt_img = preprocess_img(gt_img, binarize=False, grayscale=False, sharpened=False, blurred=False, threshold=150, vis=False)
    sr_img = preprocess_img(sr_img, binarize=False, grayscale=False, sharpened=False, blurred=False, threshold=150, vis=False)
    bounding_boxes, detected_texts = get_bounding_boxes(gt_img,vis=vis,bb_name=bb_name)
    cropped_bounding_boxes = crop_bounding_boxes(gt_img,sr_img, bounding_boxes, vis=vis)
    ocr_res = ocr_2nd_stage(detected_texts,cropped_bounding_boxes, vis=vis)
    metrics = eval_ocr_results(ocr_res, cropped_bounding_boxes)
    print("\n=== OCR Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")

        

def preprocess_img(img, binarize=False, grayscale=False,sharpened=False, blurred=False, threshold = 150, vis=False):
    '''
    Preprocess the input image by applying various transformations.
    Parameters:
        img: Input image.
        binarize: Whether to apply binarization.
        grayscale: Whether to convert to grayscale.
        sharpened: Whether to apply sharpening.
        blurred: Whether to apply blurring.
        threshold: Threshold value for binarization.
        vis: Whether to visualize the preprocessing steps.
    Returns:
        img: Preprocessed image.
    '''
    applied_steps = []
    if grayscale or binarize:
        img = img.convert('L')
        applied_steps.append("Grayscale")
    if binarize:
        img = img.point(lambda p: 255 if p > threshold else 0)
        applied_steps.append("Binarized")
    if sharpened:
        img = img.filter(ImageFilter.SHARPEN)
        applied_steps.append("Sharpened")
    if blurred:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        applied_steps.append("Blurred")
    if vis:
        plt.figure(figsize=(5, 5))
        plt.imshow(img,cmap='gray')
        title = " | ".join(applied_steps) if applied_steps else "Original Image"
        plt.title(title)
        plt.axis("off")
        plt.show()
    return img

def extract_text(img_path=None, img=None,print=False):
    '''Extract text from an image using Tesseract OCR.
    Parameters:
        img_path: Path to the image file.
        img: PIL Image object. If provided, img_path is ignored.
        print: Whether to print the extracted text.
    Returns:
        text: Extracted text string.
    '''
    if img == None:
        image = Image.open(img_path)
    else:
        image = img
    text = pytesseract.image_to_string(image)
    if print:
        print("Extracted Text:\n", text)
    return text
    
def calc_WER(gt_text, ocr_text):
    '''Calculate Word Error Rate (WER) between ground truth and OCR text.
    Parameters:
        gt_text: Ground truth text string.
        ocr_text: OCR text string.
    Returns:
        wer: Word Error Rate (WER) as a float.
    '''
    if not gt_text or not ocr_text:
        wer = 1.0
    else:
        gt_words = gt_text.split()
        ocr_words = ocr_text.split()
        wer = Levenshtein.distance(gt_words, ocr_words) / max(len(gt_words), len(ocr_words), 1)
    return wer

def calc_CER(gt_text, ocr_text):
    '''Calculate Character Error Rate (CER) between ground truth and OCR text.
    Parameters:
        gt_text: Ground truth text string.
        ocr_text: OCR text string.
    Returns:
        cer: Character Error Rate (CER) as a float.
    '''
    if not gt_text or not ocr_text:
        cer = 1.0
    else:
        cer = Levenshtein.distance(gt_text, ocr_text) / max(len(gt_text), len(ocr_text),1)
    return cer

def calc_text_coverage(gt_text, ocr_text, threshold=60):
    '''Calculate text coverage between ground truth and OCR text using fuzzy matching.
    Parameters:
        gt_text: Ground truth text string.
        ocr_text: OCR text string.
        threshold: Similarity threshold for considering a match (0-100).
    Returns:
        coverage: Text coverage as a float.
    '''
    if not gt_text or not ocr_text:
        coverage = 0.0
    else:
        gt_words = set(re.sub(r'\W+', '', word.lower()) for word in gt_text.split())
        ocr_words = set(re.sub(r'\W+', '', word.lower()) for word in ocr_text.split())
        
        match_count = 0
    
        for ocr_word in ocr_words:
            scores = [fuzz.ratio(ocr_word, gt_word) for gt_word in gt_words]
            best_score = max([fuzz.ratio(ocr_word, gt_word) for gt_word in gt_words] or [0])
            if best_score >= threshold:
                match_count += 1
        coverage = match_count / max(len(gt_words), 1)
    return coverage

def calc_classic_coverage(gt_text, ocr_text):
    '''Calculate classic text coverage between ground truth and OCR text.
    Parameters:
        gt_text: Ground truth text string.
        ocr_text: OCR text string.
    Returns:
        coverage: Classic text coverage as a float.
    '''
    gt_words = set(gt_text.split())
    ocr_words = set(ocr_text.split())
    coverage = len(gt_words.intersection(ocr_words)) / max(len(gt_words), 1)
    return coverage

def _normalize_text(text):
    '''Normalize text by lowercasing, removing punctuation, and collapsing spaces.'''
    text = re.sub(r'[^a-z0-9\s]', '', text.lower()) #remove punctuation, keep letters+digits
    return re.sub(r'\s+', ' ', text).strip() # collapse multiple spaces

def calc_soft_coverage(gt_text, ocr_text):
    '''Calculate soft text coverage between ground truth and OCR text using fuzzy matching.
    Parameters:
        gt_text: Ground truth text string.
        ocr_text: OCR text string.
    Returns:
        soft_coverage: Soft text coverage as a float.
    '''
    gt_norm  = _normalize_text(gt_text)
    ocr_norm = _normalize_text(ocr_text)

    if gt_norm == ocr_norm and gt_norm:
        return 1.0

    gt_words  = gt_norm.split()
    ocr_words = ocr_norm.split()

    if not gt_words or not ocr_words:
        return 0.0

    total_score = 0.0
    for gw in gt_words:
        best = max(fuzz.ratio(gw, ow) for ow in ocr_words)
        total_score += best / 100.0

    return total_score / len(gt_words)
