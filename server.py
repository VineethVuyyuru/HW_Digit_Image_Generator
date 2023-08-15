from flask import Flask, request, redirect, url_for, render_template
from keras.models import load_model
from numpy.random import randn
from numpy.random import randint
from matplotlib import pyplot

# Generate latent space points for generator input
def generate_latent_points(latent_dim, n_samples, target_prediction):
    # generate latent points
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)

    #generate labels
    labels = randint(target_prediction, target_prediction+1, n_samples)
    return [x_input, labels]

# Plot the generated images 
def generate_plot(images, n):
    for i in range(n*n):
        pyplot.subplot(n, n, i + 1)
        pyplot.axis('off')
        pyplot.imshow(images[i,:,:,0], cmap='gray_r')
    pyplot.savefig(r'static\\digimg.jpg')

# load model
print("Loading model...")
model = load_model('generator.h5')
latent_dim = 10
print("Model loaded.")

app = Flask(__name__)

@app.route("/", methods = ['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        inputDigit = request.form['targetDigit']

        # Check if given input is valid
        try:
            digit = int(inputDigit)
        except ValueError:
            err_msg = "Given input " + inputDigit + " is not an integer."
            return redirect(url_for('error', error_msg = err_msg))
        
        if digit < 0 or digit > 9:
            err_msg = "Given input is not in the range 0 to 9"
            print("Redirecting to error")
            return redirect(url_for('error', error_msg = err_msg))
        
        return redirect(url_for('generate', digit = digit))
        
    return render_template('index.html')

@app.route('/generate/<int:digit>')
def generate(digit):
    latent_points, label_input = generate_latent_points(latent_dim, 9, digit)
    X = model.predict([latent_points, label_input])
    generate_plot(X, 3)
    query_upper_bound = 1000000
    query_lower_bound = 0
    return render_template('generate.html', digit = digit, imgsrc = 'digimg.jpg', dummy = randint(query_lower_bound, query_upper_bound))

@app.route('/error/<error_msg>')
def error(error_msg):
    print("error:: Got the msg " + error_msg)
    return render_template('error.html', error_msg = error_msg)

app.run(host="0.0.0.0", port=5000)