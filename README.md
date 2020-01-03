# unexpected-ai-contest 

Submission Scripts for [MERCADOLIBRE DATA CHALLENGE 2019](https://ml-challenge.mercadolibre.com/) Machine Learning contest.


Final score: `0.88487` acc, position `#31` [Final results](https://ml-challenge.mercadolibre.com/final_results)

<img src="https://i.imgur.com/BpDZkmv.png" width="800">

## Notes

To Load data in parallel: `train.csv`, chuck it:
    
    mkdir train_chunks
    split -l 500000 train.csv ./train_chunks/train_chunk_ --additional-suffix=.csv

On Mac:
    
     mkdir train_chunks
     split -l 500000 train.csv ./train_chunks/train_chunk_
     cd train_chunks
     for i in *; do mv "$i" "$i.csv"; done

## Bash Launch

        nohup bash ./train_spanish_batches.sh > spanish_log.out & 
        nohup bash ./train_portuguese_batches.sh > portuguese_log.out & 

## Observations

- LSTM alone bad performance.

- LSTM + CNN works fine (1 batch 64% loss_acc time step 2600 scs): 
    
    
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(120))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

- LSTM(200) + 2xCNN performs better and faster (1 batch 66 % loss_acc time step 1500 secs )  `9_1_train_model_rich_cnn_lstm_spanish.py`:
                
                
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=64, kernel_size=16, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(200))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
                
- LSTM + 2CNN performs good in Portuguese lang. :

        
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=128, kernel_size=16, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=64, kernel_size=8, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(200))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        
- MODEL  74%,  10 epochs(Spanish) using the first batch(n=0):

    
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=256, kernel_size=8, padding='same', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        
        model.add(Dense(y_train.shape[1], activation='softmax'))
        

- *BEST* Model 80% val acc with 5 Epochs (Portuguese) using the first batch(n=0):
    
    
        model = Sequential()
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x_train.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(filters=2048, kernel_size=8, padding='same', activation='relu'))
        model.add(GlobalMaxPooling1D())
        
        model.add(Dense(2048))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        
        model.add(Dense(y_train.shape[1], activation='softmax'))        

## Requirements

- Tensorflow
- Swifter plugin for Pandas `pip install swifter`

## Pip setup

Working setup on Server `(tf-gpu) env`:
    
    absl-py==0.5.0
    astetik==1.9.5
    astor==0.7.1`
    attrs==19.1.0
    backcall==0.1.0
    beautifulsoup4==4.4.1
    bleach==2.1.4
    blessings==1.7
    bokeh==1.3.4
    certifi==2018.8.24
    chardet==3.0.4
    Click==7.0
    cloudpickle==1.2.2
    command-not-found==0.3
    cycler==0.10.0
    dask==2.4.0
    decorator==4.3.0
    defer==1.0.6
    defusedxml==0.5.0
    distributed==2.4.0
    docutils==0.14
    entrypoints==0.2.3
    fail2ban==0.9.3
    fenics-dijitso==2017.2.0
    fenics-ffc==2017.2.0
    fenics-fiat==2017.2.0
    fenics-instant==2017.2.0
    fenics-ufl==2017.2.0
    fsspec==0.4.4
    future==0.16.0
    gast==0.2.0
    geonamescache==1.0.1
    gpustat==0.5.0
    grpcio==1.15.0
    h5py==2.8.0
    HeapDict==1.0.1
    html5lib==1.0.1
    hyperas==0.4
    hyperopt==0.1.1
    ibm-cos-sdk==2.2.0
    ibm-cos-sdk-core==2.2.0
    ibm-cos-sdk-s3transfer==2.2.0
    idna==2.6
    ipykernel==4.9.0
    ipython==6.5.0
    ipython-genutils==0.2.0
    ipywidgets==7.4.2
    jedi==0.12.1
    Jinja2==2.10
    jmespath==0.9.3
    jsonschema==2.6.0
    jupyter==1.0.0
    jupyter-client==5.2.3
    jupyter-console==5.2.0
    jupyter-core==4.4.0
    Keras==2.2.2
    Keras-Applications==1.0.4
    Keras-Preprocessing==1.0.2
    kiwisolver==1.0.1
    language-selector==0.1
    llvmlite==0.29.0
    locket==0.2.0
    lxml==3.5.0
    Markdown==2.6.11
    MarkupSafe==1.0
    matplotlib==2.2.3
    mistune==0.8.3
    mpi4py==1.3.1
    mpmath==0.19
    msgpack==0.6.1
    nbconvert==5.4.0
    nbformat==4.4.0
    networkx==2.2
    notebook==5.7.0
    numba==0.45.1
    numpy==1.14.5
    nvidia-ml-py3==7.352.0
    packaging==19.1
    pandas==0.23.4
    pandocfilters==1.4.2
    parso==0.5.1
    partd==1.0.0
    patsy==0.5.0
    petsc4py==3.7.0
    pexpect==4.6.0
    pickleshare==0.7.4
    Pillow==6.1.0
    pkg-resources==0.0.0
    ply==3.7
    prometheus-client==0.3.1
    prompt-toolkit==1.0.15
    protobuf==3.6.1
    psutil==5.4.7
    ptyprocess==0.6.0
    pycups==1.9.73
    pycurl==7.43.0
    Pygments==2.2.0
    pygobject==3.20.0
    pyinotify==0.9.6
    pymongo==3.7.1
    pyparsing==2.2.1
    python-apt==1.1.0b1+ubuntu0.16.4.2
    python-dateutil==2.7.3
    python-systemd==231
    pytz==2018.5
    pyxdg==0.25
    PyYAML==3.13
    pyzmq==17.1.2
    qtconsole==4.4.1
    requests==2.18.4
    scikit-learn==0.19.2
    scipy==1.1.0
    screen-resolution-extra==0.0.0
    seaborn==0.9.0
    Send2Trash==1.5.0
    simplegeneric==0.8.1
    six==1.10.0
    slepc4py==3.7.0
    sortedcontainers==2.1.0
    ssh-import-id==5.5
    statsmodels==0.9.0
    swifter==0.295
    sympy==0.7.6.1
    system-service==0.3
    talos==0.1.9
    tblib==1.4.0
    tensorboard==1.10.0
    tensorflow-gpu==1.10.1
    termcolor==1.1.0
    terminado==0.8.1
    testpath==0.3.1
    toolz==0.10.0
    tornado==5.1.1
    tqdm==4.35.0
    traitlets==4.3.2
    ubuntu-drivers-common==0.0.0
    ufw==0.35
    unattended-upgrades==0.1
    urllib3==1.22
    virtualenv==15.0.1
    wcwidth==0.1.7
    webencodings==0.5.1
    Werkzeug==0.14.1
    widgetsnbextension==3.4.2
    xkit==0.0.0
    zict==1.0.0