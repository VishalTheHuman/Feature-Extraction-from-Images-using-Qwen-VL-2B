# Feature Extraction from Images using Qwen-VL-2B

## Contributors

<table style="width:100%; text-align:center;border: none;">
    <tr>
        <td style="width:33.33%;"><img src="https://github.com/VishalTheHuman.png/" style="width:100%; height:auto;"></td>
        <td style="width:33.33%;"><img src="https://github.com/amri-tah.png/" style="width:100%; height:auto;"></td>
        <td style="width:33.33%;"><img src="https://github.com/SaranDharshanSP.png/" style="width:100%; height:auto;"></td>
	<td style="width:33.33%;"><img src="https://github.com/SURIYA-KP.png/" style="width:100%; height:auto;"></td>
    </tr>
    <tr>
        <td><a href="https://github.com/VishalTheHuman" style="display:block; margin:auto;">@VishalTheHuman</a></td>
        <td><a href="https://github.com/amri-tah" style="display:block; margin:auto;">@amri-tah</a></td>
        <td><a href="https://github.com/SaranDharshanSP" style="display:block; margin:auto;">@SaranDharshanSP</a></td>
	<td><a href="https://github.com/SURIYA-KP" style="display:block; margin:auto;">@SURIYA-KP</a></td>
    </tr>
    <tr>
        <td><b style="display:block; margin:auto;">Vishal S</b></td>
        <td><b style="display:block; margin:auto;">Amritha Nandini</b></td>
        <td><b style="display:block; margin:auto;">Saran Dharshan S P</b></td>
	<td><b style="display:block; margin:auto;">Suriya KP</b></td>
    </tr>
</table>

## Hackathon Result

![1726992851068](image/README/1726992851068.png)

**43rd Team out of 18,000+ teams.**

## Approach

Extracting entity values from product images are crucial for sectors like healthcare, e-commerce, andcontent moderation, where product details such as weight, dimensions, wattage, and voltage are vitalfor accurate listings.  Our approach focused on utilizing theQwen2-VL-2B-Instruct model[1] [2],making use of its vision-language capabilities to interpret both visual and textual information withinimages.

**Image Pre-processing:** We resized images, ensuring they fit within the model’s input constraintsandenhanced the contrastof the images using PIL’s ImageEnhance to optimize the input qualityfor better text generation.

**Prompt Engineering:** In our approach, we crafted targeted prompts based on the product entity,such as ”Identify the height of the product” or ”Determine the wattage of the product.” These promptsweredynamically generated based on the entity name,  ensuring that the model’s focus wasdirected  towards  extracting  relevant  information  from  the  product  image  and  its  context.   If  therequired entity is not present in the image, the model must return an empty string.

**Model Inference:** The preprocessed images, along with their respective prompts, were passed intothe Qwen-2-VL-2B-Instruct model, which generated descriptive text. We used a conditional generationapproach to extract relevant information, such as weight, dimensions, or voltage.  The model was runonA6000 GPUsto ensure high computational efficiency.  We processed the images in batches of 100,saving the generated text to a CSV file after each batch to ensure incremental progress and preventdata loss.

**Post Processing:** After  obtaining  the  descriptive  text  output  from  the  model,  we  applied  postprocessing  techniques  to  convert  the  generated  text  into  the  desired  format.   The  model’s  outputtypically contained additional context, such as full sentences with product descriptions, measurements,or specifications.  To refine this into the required format,  we extracted only the relevant numericalvalues and associated units usingregular expressions.  This allowed us to isolate measurements suchas dimensions or weights.  Additionally, any abbreviated units like ”cm” or ”kg” werestandardizedinto their full forms, such as ”centimeters” or ”kilograms,” to ensure uniformity across the dataset.Once the relevant values and units were extracted, we cleaned the text by removing unnecessary wordsand maintaining only the critical data, resulting in a concise and well-formatted output.  These postprocessing steps ensured that the final output met the required specifications for our task.1

## About The Model

The Qwen-2-VL-2B-Instruct model is designed for vision-language tasks, capable of interpreting imagesand generating relevant textual descriptions based on prompts.  Itcombines a vision encoder witha language model to understand image content and generate conditioned responses.  Themodel’s ability to integrate both visual and textual data made it ideal for our product informationextraction task.

![](https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg)

## Experiments

In  our  experiments,  we  used  the  Qwen2-VL-2B-Instruct  model  for  product  information  extraction.After  several  rounds  of  optimization  and  postprocessing  of  the  test  data  provided,  we  achieved  anF1 score of 0.627.  This result highlights the model’s capability to perform extraction tasks with areasonable balance between precision and recall.

## Conclusion

In conclusion, the implementation of the model and the postprocessing pipeline allowed us to efficientlyextract and refine relevant product information from the input images.  By utilizing A6000 GPUs forfaster computation and the Qwen2-VL-2B-Instruct model, we successfully generated detailed outputsfor  various  product  attributes.   This  approach  ensured  accurate  extraction  of  product  dimensions,weights, and other specifications, ultimately enhancing the usability and consistency of the data forfurther analysis or integration into other systems.References[1]  Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, ChangZhou, and Jingren Zhou.  Qwen-vl:  A versatile vision-language model for understanding, localiza-tion, text reading, and beyond.arXiv preprint arXiv:2308.12966, 2023.[2]  Qwen team.  Qwen2-vl.  2024.2

# Amazon ML Challenge Problem Statement

## Feature Extraction from Images

In this hackathon, the goal is to create a machine learning model that extracts entity values from images. This capability is crucial in fields like healthcare, e-commerce, and content moderation, where precise product information is vital. As digital marketplaces expand, many products lack detailed textual descriptions, making it essential to obtain key details directly from images. These images provide important information such as weight, volume, voltage, wattage, dimensions, and many more, which are critical for digital stores.

### Data Description:

The dataset consists of the following columns:

1. **index:** An unique identifier (ID) for the data sample
2. **image_link**: Public URL where the product image is available for download. Example link - https://m.media-amazon.com/images/I/71XfHPR36-L.jpg
   To download images use `download_images` function from `src/utils.py`. See sample code in `src/test.ipynb`.
3. **group_id**: Category code of the product
4. **entity_name:** Product entity name. For eg: “item_weight”
5. **entity_value:** Product entity value. For eg: “34 gram”
   Note: For test.csv, you will not see the column `entity_value` as it is the target variable.

### Output Format:

The output file should be a csv with 2 columns:

1. **index:** The unique identifier (ID) of the data sample. Note the index should match the test record index.
2. **prediction:** A string which should have the following format: “x unit” where x is a float number in standard formatting and unit is one of the allowed units (allowed units are mentioned in the Appendix). The two values should be concatenated and have a space between them. For eg: “2 gram”, “12.5 centimetre”, “2.56 ounce” are valid. Few invalid cases: “2 gms”, “60 ounce/1.7 kilogram”, “2.2e2 kilogram” etc.
   Note: Make sure to output a prediction for all indices. If no value is found in the image for any test sample, return empty string, i.e, `“”`. If you have less/more number of output samples in the output file as compared to test.csv, your output won’t be evaluated.

### File Descriptions:

*source files*

1. **src/sanity.py**: Sanity checker to ensure that the final output file passes all formatting checks. Note: the script will not check if less/more number of predictions are present compared to the test file. See sample code in `src/test.ipynb`
2. **src/utils.py**: Contains helper functions for downloading images from the image_link.
3. **src/constants.py:** Contains the allowed units for each entity type.
4. **sample_code.py:** We also provided a sample dummy code that can generate an output file in the given format. Usage of this file is optional.

*Dataset files*

1. **dataset/train.csv**: Training file with labels (`entity_value`).
2. **dataset/test.csv**: Test file without output labels (`entity_value`). Generate predictions using your model/solution on this file's data and format the output file to match sample_test_out.csv (Refer the above section "Output Format")
3. **dataset/sample_test.csv**: Sample test input file.
4. **dataset/sample_test_out.csv**: Sample outputs for sample_test.csv. The output for test.csv must be formatted in the exact same way. Note: The predictions in the file might not be correct

### Constraints

1. You will be provided with a sample output file and a sanity checker file. Format your output to match the sample output file exactly and pass it through the sanity checker to ensure its validity. Note: If the file does not pass through the sanity checker, it will not be evaluated. You should recieve a message like `Parsing successfull for file: ...csv` if the output file is correctly formatted.
2. You are given the list of allowed units in constants.py and also in Appendix. Your outputs must be in these units. Predictions using any other units will be considered invalid during validation.

### Evaluation Criteria

Submissions will be evaluated based on F1 score, which are standard measures of prediction accuracy for classification and extraction problems.

Let GT = Ground truth value for a sample and OUT be output prediction from the model for a sample. Then we classify the predictions into one of the 4 classes with the following logic:

1. *True Positives* - If OUT != `""` and GT != `""` and OUT == GT
2. *False Positives* - If OUT != `""` and GT != `""` and OUT != GT
3. *False Positives* - If OUT != `""` and GT == `""`
4. *False Negatives* - If OUT == `""` and GT != `""`
5. *True Negatives* - If OUT == `""` and GT == `""`

Then, F1 score = 2*Precision*Recall/(Precision + Recall) where:

1. Precision = True Positives/(True Positives + False Positives)
2. Recall = True Positives/(True Positives + False Negatives)

### Submission File

Upload a test_out.csv file in the Portal with the exact same formatting as sample_test_out.csv

### Appendix

```
entity_unit_map = {
  "width": {
    "centimetre",
    "foot",
    "millimetre",
    "metre",
    "inch",
    "yard"
  },
  "depth": {
    "centimetre",
    "foot",
    "millimetre",
    "metre",
    "inch",
    "yard"
  },
  "height": {
    "centimetre",
    "foot",
    "millimetre",
    "metre",
    "inch",
    "yard"
  },
  "item_weight": {
    "milligram",
    "kilogram",
    "microgram",
    "gram",
    "ounce",
    "ton",
    "pound"
  },
  "maximum_weight_recommendation": {
    "milligram",
    "kilogram",
    "microgram",
    "gram",
    "ounce",
    "ton",
    "pound"
  },
  "voltage": {
    "millivolt",
    "kilovolt",
    "volt"
  },
  "wattage": {
    "kilowatt",
    "watt"
  },
  "item_volume": {
    "cubic foot",
    "microlitre",
    "cup",
    "fluid ounce",
    "centilitre",
    "imperial gallon",
    "pint",
    "decilitre",
    "litre",
    "millilitre",
    "quart",
    "cubic inch",
    "gallon"
  }
}
```
