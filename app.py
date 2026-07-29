# Python 3.11, Tensorflow 2.12.0
import os
from dotenv import load_dotenv
import requests
from groq import Groq
try:
    from tf_keras.models import load_model
except ImportError:
    from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, render_template, redirect, url_for, request, session
from forms import uploadForm

load_dotenv()
searchApiKey = os.getenv('SEARCH_API_KEY')
searchEngineId = os.getenv('SEARCH_ENGINE_ID')
client = Groq(
    api_key=os.getenv('GROQ_API_KEY')
)
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
search = True

@app.route('/', methods=['GET'])
def about(): #summaries for each disease
    farmCrops = [ 
        createLLMResponse("Banana Furasium Wilt", 0),
        createLLMResponse("Wheat Leaf Blight", 0),
        createLLMResponse("Potato Blight - Early", 0),
        createLLMResponse("Potato Blight - Late", 0),
        createLLMResponse("Mango Anthracnose", 0),
        createLLMResponse("Cotton Bacterial Blight", 0),
        createLLMResponse("Rice Leaf Smut", 0),
        createLLMResponse("Potato Nematode", 0),
        createLLMResponse("Tomato Leaf Mold", 0),
        createLLMResponse("Strawberry Leaf Scorch", 0),
        createLLMResponse("Corn Common Rust", 0),
        createLLMResponse("Corn Blight", 0),
        createLLMResponse("Bell Pepper Bacterial Spot", 0),
        createLLMResponse("Grape Black Measles", 0),
        createLLMResponse("Tomato Two-spotted spider mite", 0)
    ]
    
    homePlants = [
        createLLMResponse("Scorch", 0),
        createLLMResponse("Powdery Mildew", 0),
        createLLMResponse("Rust Plant Disease", 0),
        createLLMResponse("Hibiscus Blight", 0),
        createLLMResponse("Red Spot", 0),
        createLLMResponse("Blight", 0)
        
    ]
    return render_template('about.html', farmCrops=farmCrops, homePlants=homePlants)

@app.route('/submit/<int:type>', methods=['GET', 'POST'])
def submit(type):
    form = uploadForm()
    if request.method == 'POST':
        
        user_upload = request.files['imageUpload']
        filename = user_upload.filename
        if not (len(filename) > 0): # verifying file submission
            return render_template('submit.html', form=form)
        
        image_path = "./static/user_uploads/" + filename
        user_upload.save(image_path)
        
        prediction = predict(image_path, type)
        disease=prediction[0]
        summary=createLLMResponse(disease, 1)
        
        website_links=[]
        if search:
            website_links=getCredibleSources(disease)
        session['prediction'] = prediction
        session['img_path'] = image_path
        session['name'] = filename
        session['website_links'] = website_links
        session['summary'] = summary
        
        return redirect(url_for('result'))
    
    return render_template('submit.html', form=form, type=type)


@app.route('/result', methods=['GET'])
def result():
    prediction=None
    img_path=None
    name=None
    website_links=None
    summary=None
    if 'prediction' in session:
        prediction=session['prediction']
    else:
        return redirect(url_for('index')) # blank result
    if 'img_path' in session:
        img_path=session['img_path']
    if 'name' in session:
        name=session['name']
    if 'website_links' in session:
        website_links=session['website_links']
    if 'summary' in session:
        summary=session['summary']
    return render_template('result.html', prediction=prediction, img_path=img_path, name=name, website_links=website_links, summary=summary)

FALLBACK_SUMMARIES_LONG = {
    "Rice Leaf Smut": "Rice Leaf Smut is a fungal infection caused by <i>Entyloma oryzae</i> that creates small, dark linear spots on mature rice leaves. While typically mild, active management ensures maximum grain quality and crop yield.<br><br><b>Recommended Treatment & Care Steps:</b><ul><li><b>Plant Disease-Resistant Varieties:</b> Select certified resistant seed cultivars optimal for your growing zone.</li><li><b>Balanced Nitrogen Fertilizer:</b> Avoid over-fertilization with nitrogen, which promotes lush growth susceptible to fungal spread.</li><li><b>Fungicidal Control:</b> Apply preventive copper-based or triazole fungicides during early tillering if leaf spots appear.</li><li><b>Crop Rotation & Sanitation:</b> Plow under or remove post-harvest crop residue to eliminate overwintering spores.</li></ul>",
    "Potato Blight - Early": "Early Blight in potatoes is caused by the fungus <i>Alternaria solani</i>, producing dark concentric ring spots on older leaves.<br><br><b>Recommended Treatment & Care Steps:</b><ul><li><b>Fungicide Application:</b> Apply chlorothalonil or copper fungicides at the first sign of leaf spots.</li><li><b>Proper Irrigation:</b> Water at the soil base using drip irrigation to avoid wet foliage.</li><li><b>Crop Destruction:</b> Remove infected foliage before harvest to protect tubers.</li></ul>",
    "Tomato Leaf Mold": "Tomato Leaf Mold is caused by <i>Passalora fulva</i>, leading to pale green or yellow spots on upper leaf surfaces.<br><br><b>Recommended Treatment & Care Steps:</b><ul><li><b>Improve Ventilation:</b> Increase greenhouse airflow and lower relative humidity below 85%.</li><li><b>Foliar Fungicide:</b> Use protective fungicides containing copper octanoate or difenoconazole.</li><li><b>Pruning:</b> Trim lower infected leaves to enhance air circulation.</li></ul>",
}

FALLBACK_SUMMARIES_SHORT = {
    "Rice Leaf Smut": "Rice Leaf Smut is a fungal disease causing small black linear spots on rice leaves, managed effectively with balanced fertilization and resistant varieties.",
    "Potato Blight - Early": "Early Blight produces target-like concentric spots on potato leaves and can be controlled with crop rotation and protective fungicides.",
    "Tomato Leaf Mold": "Tomato Leaf Mold causes yellow spots and velvety mold on tomato leaves, favored by high humidity and poor ventilation.",
}

def createLLMResponse(inp, num):
    inp = inp.strip()
    if num == 1: #lengthy response about how to cure
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                    "role": "system",
                    "content": "you are a helpful assistant on plants and a helpful assistant on plant diseases."
                    },
                    {
                        "role": "user",
                        "content": "In your response, do include any messages such as 'Ok. Here's a response:' or 'Sure! Here's some info about that:'. Only answer the question directly, and answer in a positive tone. The user is asking about information about " + inp + " and how to cure " + inp + ". DO NOT include any single or double asteriks in your response. Surround anything that's supposed to be in bold with HTML bold tags and surround anything that's supposed to be in italics with HTML italic tags. Surround any list items with <ul> tags and surround individual bullet points in <li> tags"
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            return completion.choices[0].message.content
        except:
            return FALLBACK_SUMMARIES_LONG.get(
                inp,
                f"{inp} is a common plant condition that affects plant vigor and foliage.<br><br><b>Recommended Treatment & Care Steps:</b><ul><li><b>Prune Affected Leaves:</b> Trim infected foliage to stop disease progression.</li><li><b>Apply Protective Spray:</b> Treat leaves with copper-based organic fungicides or neem oil.</li><li><b>Optimize Soil Hydration:</b> Water at the root zone to keep foliage dry.</li></ul>"
            )
    elif num == 0: #short info response
        try:
            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                    "role": "system",
                    "content": "you are a helpful assistant on plants and a helpful assistant on plant diseases."
                    },
                    {
                        "role": "user",
                        "content": "In your response, do include any messages such as 'Ok. Here's a response:' or 'Sure! Here's some info about that:'. Only answer the question directly, and answer in a positive tone. The user is asking about a two sentence information about " + inp + ". DO NOT include any single or double asteriks in your response."
                    }
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            return completion.choices[0].message.content
        except:
            return FALLBACK_SUMMARIES_SHORT.get(
                inp,
                f"{inp} is a plant disease that impacts crop foliage and health, manageable through proper crop sanitation and targeted plant care."
            )


def getCredibleSources(query):
    try:
        url = 'https://www.googleapis.com/customsearch/v1'
        search_params = {
            'q': ("How to cure " + query), # the words it searches
            'key': searchApiKey, # api key for authentication
            'cx' : searchEngineId, # search engine to use
            'num' : 5, #number of results
            'siteSearch' : 'edu',
        }
        response = requests.get(url, params=search_params)
        results = response.json().get('items', [])
        
        links = []
        for result in results:
            links.append(result['link']) # accesses the link
            
        if not links:
            links = [
                "https://extension.psu.edu/plant-disease-identification-and-management",
                "https://ipm.ucanr.edu/agriculture/plant-health-guide/",
                "https://edis.ifas.ufl.edu/publication/AG312"
            ]
        return links
    except Exception:
        return [
            "https://extension.psu.edu/plant-disease-identification-and-management",
            "https://ipm.ucanr.edu/agriculture/plant-health-guide/",
            "https://edis.ifas.ufl.edu/publication/AG312"
        ]

def predict(path, type):
    
    np.set_printoptions(suppress=True) # disable scientific notation

    if type == 1: #farm crops
        model = load_model("keras_model.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()
    elif type == 0: #home plants
        model = load_model("keras_model_house.h5", compile=False)
        class_names = open("labels_house.txt", "r").readlines()

    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # create the array of the correct size

    image = Image.open(path).convert("RGB")

    # resizing the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    
    image_array = np.asarray(image) # image being converted to numpy array

    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1 #normalizing image

    data[0] = normalized_image_array #load image into array

    # model prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    return [class_name[2:].strip(), round(float(confidence_score) * 100, 2)] #class and confidence_score

if __name__ == '__main__':
    app.run(debug=True) #runs the app