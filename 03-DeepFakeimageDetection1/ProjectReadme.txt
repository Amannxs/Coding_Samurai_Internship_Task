Deepfake Image Detector
A PyTorch-based tool that tells real human photos apart from AI-generated fakes. Comes with a simple Streamlit interface where you can drag and drop images for instant analysis.

What I Was Working On Before This
I'd just finished a customer churn project where I hit 99.8% validation with LightGBM, but I wanted to switch gears and tackle something in computer vision. This felt like a fun challenge - figuring out if an image is real or AI-generated.

Problems I Actually Ran Into
The Dataset Was Too Small
The Kaggle dataset only had about 1,000 images total (436 real, 547 fake). My first thought was to try semi-supervised learning, but that would've been a disaster with this little data - the model would just reinforce its own wrong guesses. Instead, I went with transfer learning (EfficientNet-B4) and threw heavy data augmentation at it - rotations, flips, color changes - to make the most of what I had.

Annoying PIL Warnings
During training, PyTorch kept complaining: Palette images with Transparency expressed in bytes should be converted to RGBA images. Some PNGs had weird transparency settings that didn't play nice with RGB format. I fixed this by adding a check in my __getitem__ method that catches RGBA, LA, or palette mode images and converts them properly before they hit the transformation pipeline.

Indentation Errors That Drove Me Nuts
Python threw an IndentationError at line 95 and some random t32 typo that broke my DataLoader print statement. Had to go through and clean up the formatting in my custom dataset class - nothing major, just one of those things that stops you dead for five minutes.

Streamlit Page Config Timing
The Streamlit app kept crashing with StreamlitSetPageConfigMustBeFirstCommandError. Turns out I had an error handler running in a cached model loader that was executing before st.set_page_config(). Moved the page config to the very top of app.py right after imports and it worked fine.

How the Model Works
Architecture: EfficientNet-B4 from torchvision. Picked this because it's great at picking up fine details - skin textures, edge blending, pixel-level artifacts that give away fake images.

Input Size: 380×380 pixels (required for EfficientNet-B4)

Final Layer: Replaced the default 1000-class classifier with a Dropout layer (p=0.4) and a single output neuron for binary classification. Using BCEWithLogitsLoss() as the loss function.

Performance: Hit about 87.82% validation accuracy after 10 epochs. It catches fake images with over 99.9% confidence on the test set.

Running It on Your Machine
Make sure you have Python 3.10+ installed

Download the model weights file (best_deepfake_b4_model.pth) from Kaggle and drop it in the project folder

Open VS Code terminal and install dependencies:

bash
pip install -r requirements.txt
Launch the app:

bash
streamlit run app.py
Upload an image and see if it's real or fake

