# muski-performance-rnn

Adaptation of the 
[Google Magenta Performance RNN Browser demo](https://github.com/magenta/magenta-demos/tree/main/performance_rnn) 
for the Muski website.

Adapted from the port of the [Performance RNN](https://magenta.tensorflow.org/performance-rnn) model
to the [TensorFlow.js](https://js.tensorflow.org) environment by Google Magenta.

# Building

To build, execute `yarn bundle`. This will regenerate `bundle.js`, which is referenced by `index.html`.

To view, execute `yarn run-demo`, which will start a local webserver.

# Credits

Original demo by [Ian Simon](https://github.com/iansimon) and [Sageev Oore](https://github.com/osageev).

Google.

Adapted for the Muski website by Eric Londaits, for IMAGINARY gGmbH.

[Performance RNN](https://magenta.tensorflow.org/performance-rnn) was trained in TensorFlow on 
MIDI from piano performances from the 
[Yamaha e-Piano Competition dataset](http://www.piano-e-competition.com/). 
It was then ported to run in the browser using only Javascript in the 
[TensorFlow.js](https://js.tensorflow.org/) environment. Piano samples are from 
[Salamander Grand Piano](https://archive.org/details/SalamanderGrandPianoV3).

# License

Licensed under the Apache License, Version 2.0.

Original demo ©2017 Google.
Adaptation ©2022 IMAGINARY gGmbH.

[Salamander Grand Piano V3](https://archive.org/details/SalamanderGrandPianoV3) 
by [Alexander Holm](https://archive.org/search.php?query=creator%3A%22Alexander+Holm%22) 
licensed under [CC-BY v3.0](http://creativecommons.org/licenses/by/3.0/).
