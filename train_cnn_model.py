# Load the OCR model
model_path = '/content/drive/MyDrive/Hnd/cnn_model3.keras'
model = load_model(model_path)


# Define Kannada character array
kannada_array = [
"ಅ", "ಅಂ", "ಅಃ", "ಆ", "ಇ", "ಈ", "ಉ", "ಊ", "ಋ", "ಎ", "ಏ", "ಐ", "ಒ", "ಓ", "ಔ",
"ಕ", "ಕಂ", "ಕಃ", "ಕಾ", "ಕಿ", "ಕೀ", "ಕು", "ಕೂ", "ಕೃ", "ಕೄ", "ಕೆ", "ಕೊ", "ಕೇ", "ಕೈ", "ಕೋ", "ಕೌ", "ಕ್",
"ಕ್ಷ", "ಕ್ಷಂ", "ಕ್ಷಃ", "ಕ್ಷಾ", "ಕ್ಷಿ", "ಕ್ಷೀ", "ಕ್ಷು", "ಕ್ಷೂ", "ಕ್ಷೃ", "ಕ್ಷೆ", "ಕ್ಷೇ", "ಕ್ಷೆ", "ಕ್ಷೊ", "ಕ್ಷೋ", "ಕ್ಷೌ", "ಕ್ಷ್",
"ಖಂ", "ಖಃ", "ಖಾ", "ಖಿ", "ಖೀ", "ಖು", "ಖೂ", "ಖೃ", "ಖೆ", "ಖೇ", "ಖೊ", "ಖೋ", "ಖೌ", "ಖ್", "ಖ್ಯ", "ಖ್ಯೆ",
"ಗ", "ಗಂ", "ಗಃ", "ಗಾ", "ಗಿ", "ಗೀ", "ಗು", "ಗೂ", "ಗೃ", "ಗೆ", "ಗೇ", "ಗೈ", "ಗೊ", "ಗೋ", "ಗೌ", "ಗ್",
"ಘ", "ಘಂ", "ಘಃ", "ಘಾ", "ಘಿ", "ಘೀ", "ಘು", "ಘೂ", "ಘೃ", "ಘೇ", "ಘೈ", "ಘೊ", "ಘೋ", "ಘೌ", "ಘ್",
"ಙ", "ಙಂ", "ಙಃ", "ಙಾ", "ಙಿ", "ಙೀ", "ಙು", "ಙೂ", "ಙೃ", "ಙೆ", "ಙೀ", "ಙೇ", "ಙೊ", "ಙೋ", "ಙೌ", "ಙ್",
"ಚ", "ಚಂ", "ಚಃ", "ಚಾ", "ಚಿ", "ಚೀ", "ಚು", "ಚೂ", "ಚೃ", "ಚೆ", "ಚೇ", "ಚೈ", "ಚೊ", "ಚೋ", "ಚೌ", "ಚ್",
"ಛ", "ಛಂ", "ಛಃ", "ಛಾ", "ಛಿ", "ಛೀ", "ಛು", "ಛೂ", "ಛೃ", "ಛೆ", "ಛೇ", "ಛೈ", "ಛೊ", "ಛೋ", "ಛೌ", "ಛ್",
"ಜ", "ಜಂ", "ಜಃ", "ಜಾ", "ಜಿ", "ಜೀ", "ಜು", "ಜೂ", "ಜೃ", "ಜೆ", "ಜೇ", "ಜೈ", "ಜೊ", "ಜೋ", "ಜೌ", "ಜ್",
"ಝ", "ಝಂ", "ಝಃ", "ಝಾ", "ಝಿ", "ಝೀ", "ಝು", "ಝೂ", "ಝೃ", "ಝೆ", "ಝೇ", "ಝೈ", "ಝೊ", "ಝೋ", "ಝೌ", "ಝ್",
"ಞ", "ಞಂ", "ಞಃ", "ಞಾ", "ಞಿ", "ಞೀ", "ಞು", "ಞೂ", "ಞೃ", "ಞೆ", "ಞೇ", "ಞೈ", "ಞೊ", "ಞೋ", "ಞೌ", "ಞ್",
"ಟ", "ಟಂ", "ಟಃ", "ಟಾ", "ಟಿ", "ಟೀ", "ಟು", "ಟೂ", "ಟೃ", "ಟೆ", "ಟೇ", "ಟೈ", "ಟೊ", "ಟೋ", "ಟೌ", "ಟ್",
"ಠ", "ಠಂ", "ಠಃ", "ಠಾ", "ಠಿ", "ಠೀ", "ಠು", "ಠೂ", "ಠೃ", "ಠೆ", "ಠೇ", "ಠೈ", "ಠೊ", "ಠೋ", "ಠೌ", "ಠ್",
"ಡ", "ಡಂ", "ಡಃ", "ಡಾ", "ಡಿ", "ಡೀ", "ಡು", "ಡೂ", "ಡೃ", "ಡೆ", "ಡೇ", "ಡೈ", "ಡೊ", "ಡೋ", "ಡೌ", "ಡ್",
"ಢ", "ಢಂ", "ಢಃ", "ಢಾ", "ಢಿ", "ಢೀ", "ಢು", "ಢೂ", "ಢೃ", "ಢೆ", "ಢೇ", "ಢೈ", "ಢೊ", "ಢೋ", "ಢೌ", "ಢ್",
"ಣ", "ಣಂ", "ಣಃ", "ಣಾ", "ಣಿ", "ಣೀ", "ಣು", "ಣೂ", "ಣೃ", "ಣೆ", "ಣೇ", "ಣೈ", "ಣೊ", "ಣೋ", "ಣೌ", "ಣ್",
"ತ", "ತಂ", "ತಃ", "ತಾ", "ತಿ", "ತೀ", "ತು", "ತೂ", "ತೃ", "ತೆ", "ತೇ", "ತೈ", "ತೊ", "ತೋ", "ತೌ", "ತ್",
"ಥ", "ಥಂ", "ಥಃ", "ಥಾ", "ಥಿ", "ಥೀ", "ಥು", "ಥೂ", "ಥೃ", "ಥೆ", "ಥೇ", "ಥೈ", "ಥೊ", "ಥೋ", "ಥೌ", "ಥ್",
"ದ", "ದಂ", "ದಃ", "ದಾ", "ದಿ", "ದೀ", "ದು", "ದೂ", "ದೃ", "ದೆ", "ದೇ", "ದೈ", "ದೊ", "ದೋ", "ದೌ", "ದ್",
"ಧ", "ಧಂ", "ಧಃ", "ಧಾ", "ಧಿ", "ಧೀ", "ಧು", "ಧೂ", "ಧೃ", "ಧೇ", "ಧೈ", "ಧೊ", "ಧೋ", "ಧೌ", "ಧ್",
"ನ", "ನಂ", "ನಃ", "ನಾ", "ನಿ", "ನೀ", "ನು", "ನೂ", "ನೃ", "ನೆ", "ನೇ", "ನೈ", "ನೊ", "ನೋ", "ನೌ", "ನ್",
"ಪ", "ಪಂ", "ಪಃ", "ಪಾ", "ಪಿ", "ಪೀ", "ಪು", "ಪೂ", "ಪೃ", "ಪೆ", "ಪೇ", "ಪೈ", "ಪೊ", "ಪೋ", "ಪೌ", "ಪ್",
"ಫ", "ಫಂ", "ಫಃ", "ಫಾ", "ಫಿ", "ಫೀ", "ಫು", "ಫೂ", "ಫೃ", "ಫೆ", "ಫೇ", "ಫೈ", "ಫೊ", "ಫೋ", "ಫೌ", "ಫ್",
"ಬ", "ಬಂ", "ಬಃ", "ಬಾ", "ಬಿ", "ಬೀ", "ಬು", "ಬೂ", "ಬೃ", "ಬೆ", "ಬೇ", "ಬೈ", "ಬೊ", "ಬೋ", "ಬೌ", "ಬ್",
"ಭ", "ಭಂ", "ಭಃ", "ಭಾ", "ಭಿ", "ಭೀ", "ಭು", "ಭೂ", "ಭೃ", "ಭೆ", "ಭೇ", "ಭೈ", "ಭೊ", "ಭೋ", "ಭೌ", "ಭ್",
"ಮ", "ಮಂ", "ಮಃ", "ಮಾ", "ಮಿ", "ಮೀ", "ಮು", "ಮೂ", "ಮೃ", "ಮೆ", "ಮೇ", "ಮೈ", "ಮೊ", "ಮೋ", "ಮೌ", "ಮ್",
"ಯ", "ಯಂ", "ಯಃ", "ಯಾ", "ಯಿ", "ಯೀ", "ಯು", "ಯೂ", "ಯೃ", "ಯೆ", "ಯೇ", "ಯೈ", "ಯೊ", "ಯೋ", "ಯೌ", "ಯ್",
"ರ", "ರಂ", "ರಃ", "ರಾ", "ರಿ", "ರೀ", "ರು", "ರೂ", "ರೃ", "ರೆ", "ರೇ", "ರೈ", "ರೊ", "ರೋ", "ರೌ", "ರ್",
"ಲ", "ಲಂ", "ಲಃ", "ಲಾ", "ಲಿ", "ಲೀ", "ಲು", "ಲೂ", "ಲೃ", "ಲೆ", "ಲೇ", "ಲೈ", "ಲೊ", "ಲೋ", "ಲೌ", "ಲ್",
"ಳ", "ಳಂ", "ಳಃ", "ಳಾ", "ಳಿ", "ಳೀ", "ಳು", "ಳೂ", "ಳೃ", "ಳೆ", "ಳೇ", "ಳೈ", "ಳೊ", "ಳೋ", "ಳೌ", "ಳ್",
"ವ", "ವಂ", "ವಃ", "ವಾ", "ವಿ", "ವೀ", "ವು", "ವೂ", "ವೃ", "ವೆ", "ವೇ", "ವೈ", "ವೊ", "ವೋ", "ವೌ", "ವ್",
"ಶ", "ಶಂ", "ಶಃ", "ಶಾ", "ಶಿ", "ಶೀ", "ಶು", "ಶೂ", "ಶೃ", "ಶೆ", "ಶೇ", "ಶೈ", "ಶೊ", "ಶೋ", "ಶೌ", "ಶ್",
"ಷ", "ಷಂ", "ಷಃ", "ಷಾ", "ಷಿ", "ಷೀ", "ಷು", "ಷೂ", "ಷೃ", "ಷೆ", "ಷೇ", "ಷೈ", "ಷೊ", "ಷೋ", "ಷೌ", "ಷ್",
"ಸ", "ಸಂ", "ಸಃ", "ಸಾ", "ಸಿ", "ಸೀ", "ಸು", "ಸೂ", "ಸೃ", "ಸೆ", "ಸೇ", "ಸೈ", "ಸೊ", "ಸೋ", "ಸೌ", "ಸ್",
"ಹ", "ಹಂ", "ಹಃ", "ಹಾ", "ಹಿ", "ಹೀ", "ಹು", "ಹೂ", "ಹೃ", "ಹೆ", "ಹೇ", "ಹೈ", "ಹೊ", "ಹೋ", "ಹೌ", "ಹ್",
"೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"
]


def segment_lines(binary_image):
    # Sum the pixel values along the x-axis to get horizontal projection
    horizontal_projection = np.sum(binary_image, axis=1)
    # Detect line regions based on horizontal projection
    lines = []
    start = None
    for i, value in enumerate(horizontal_projection):
        if value > 0 and start is None:
            start = i
        elif value == 0 and start is not None:
            lines.append((start, i))
            start = None
    if start is not None:
        lines.append((start, len(horizontal_projection)))
    return lines


def segment_characters(binary_line_image):
    # Find contours in the line image
    contours, _ = cv2.findContours(binary_line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter contours based on area
    min_contour_area = 50
    max_contour_area = 50000
    filtered_contours = [contour for contour in contours if min_contour_area < cv2.contourArea(contour) < max_contour_area]
    # Sort contours by their bounding box x-coordinate
    filtered_contours = sorted(filtered_contours, key=lambda contour: cv2.boundingRect(contour)[0])
    # Segmented characters
    char_images = []
    bounding_boxes = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        char_image = binary_line_image[y:y+h, x:x+w]
        # Add padding with white background
        padded_char_image = cv2.copyMakeBorder(char_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        char_images.append(padded_char_image)
        bounding_boxes.append((x, y, w, h))
    return char_images, bounding_boxes

def segment_text(binary_image):
    # Segment lines
    lines = segment_lines(binary_image)
    paragraph = []
    for start, end in lines:
    # Extract the line image
        line_image = binary_image[start:end, :]
        # Segment characters within the line
        char_images, bounding_boxes = segment_characters(line_image)
        # Store characters and their positions
        line_text = {'line_image': line_image, 'char_images': char_images, 'bounding_boxes': bounding_boxes}
        paragraph.append(line_text)
    return paragraph

def preprocess_image(image_path):
    #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img=image_path
    if img is None:
        raise ValueError(f"Error reading image at {image_path}")
    img = cv2.resize(img, (128, 128))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=-1) # Add channel dimension
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img

def preprocess(image_path):
    # fetch image
    #img = cv2.imread(image_path)
    img = image_path
    #cv2_imshow(img)
    # enlarge the image
    #img = cv2.GaussianBlur(img, (3,3), 0)
    scale_factor = 2
    larger_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # gray scale and binary
    #gray = cv2.cvtColor(larger_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(larger_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # padding
    #binary = np.zeros((100, 100), dtype=np.uint8) # Example binary image
    height, width = binary.shape[:2]

    # Target dimensions
    target_height = 200
    target_width = 200

    # Calculate required padding for each side
    top_padding = (target_height - height) // 2
    bottom_padding = target_height - height - top_padding
    left_padding = (target_width - width) // 2
    right_padding = target_width - width - left_padding

    # Apply the padding
    result = cv2.copyMakeBorder(binary, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 255))

    # thinning
    kernel = np.ones((3, 3), np.uint8) # Adjust the kernel size to control the thickness (5,5)
    dilated = cv2.dilate(result, kernel, iterations=1)
    thinned = cv2.ximgproc.thinning(dilated)
    thinned_inverted = cv2.bitwise_not(thinned) #invert thinning
    img = thinned_inverted

    #thickening
    kernel = np.ones((5, 5), np.uint8) #(5,5)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    #cv2_imshow(img_erosion)

    return img_erosion


# Example usage
binary_image_path = '/content/drive/MyDrive/Hnd/images/s1.jpg'
binary_image = character_segmentation(binary_image_path)

paragraph = segment_text(binary_image)

predicted = ''
# Display the segmented characters line-by-line
for line_idx, line in enumerate(paragraph):
    #print(f"Line {line_idx + 1}:")
    predicted_line = ""
    for char_idx, char_image in enumerate(line['char_images']):
        #print(f"Character {char_idx + 1}:")
        _, binary1 = cv2.threshold(char_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #cv2_imshow(binary1) # Use cv2_imshow instead of cv2.imshow

        # Predict character using the OCR model
        # Add batch dimension
        new_image = preprocess(binary1)
        new_image = preprocess_image(new_image)

        prediction = model.predict(new_image)
        predicted_class = np.argmax(prediction, axis=1)

        # Print the predicted class
        l = predicted_class[0]
        k = kannada_array[l - 1]
        #print(f'Predicted character: {k}')
        predicted_line += k

    #print(predicted_line)
    predicted += predicted_line + '\n'

print('kannada text : ')
print(predicted)
