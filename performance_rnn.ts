/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import * as tf from '@tensorflow/tfjs-core';
import {KeyboardElement} from './keyboard_element';

// C Major scale.
const DEFAULT_PITCH_WEIGHTS = [2, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1];

// tslint:disable-next-line:no-require-imports
const Piano = require('tone-piano').Piano;

let lstmKernel1: tf.Tensor2D;
let lstmBias1: tf.Tensor1D;
let lstmKernel2: tf.Tensor2D;
let lstmBias2: tf.Tensor1D;
let lstmKernel3: tf.Tensor2D;
let lstmBias3: tf.Tensor1D;
let c: tf.Tensor2D[];
let h: tf.Tensor2D[];
let fcB: tf.Tensor1D;
let fcW: tf.Tensor2D;
const forgetBias = tf.scalar(1.0);
const activeNotes = new Map<number, number>();

let stepTimeout: NodeJS.Timer = null;
let resetTimeout: NodeJS.Timer = null;

// How many steps to generate per generateStep call.
// Generating more steps makes it less likely that we'll lag behind in note
// generation. Generating fewer steps makes it less likely that the browser UI
// thread will be starved for cycles.
const STEPS_PER_GENERATE_CALL = 10;
// How much time to try to generate ahead. More time means fewer buffer
// underruns, but also makes the lag from UI change to output larger.
const GENERATION_BUFFER_SECONDS = .5;
// If we're this far behind, reset currentTime time to piano.now().
const MAX_GENERATION_LAG_SECONDS = 1;
// If a note is held longer than this, release it.
const MAX_NOTE_DURATION_SECONDS = 3;

const NOTES_PER_OCTAVE = 12;
const DENSITY_BIN_RANGES = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
const PITCH_WEIGHT_SIZE = NOTES_PER_OCTAVE;

const RESET_RNN_FREQUENCY_MS = 30000;

let pitchDistribution: tf.Tensor1D;
let noteDensityEncoding: tf.Tensor1D;

let currentPianoTimeSec = 0;
let currentVelocity = 100;

const MIN_MIDI_PITCH = 0;
const MAX_MIDI_PITCH = 127;
const VELOCITY_BINS = 32;
const MAX_SHIFT_STEPS = 100;
const STEPS_PER_SECOND = 100;

// The unique id of the currently scheduled setTimeout loop.
let currentLoopId = 0;

const EVENT_RANGES = [
    ['note_on', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
    ['note_off', MIN_MIDI_PITCH, MAX_MIDI_PITCH],
    ['time_shift', 1, MAX_SHIFT_STEPS],
    ['velocity_change', 1, VELOCITY_BINS],
];

function calculateEventSize(): number {
    let eventOffset = 0;
    for (const eventRange of EVENT_RANGES) {
        const minValue = eventRange[1] as number;
        const maxValue = eventRange[2] as number;
        eventOffset += maxValue - minValue + 1;
    }
    return eventOffset;
}

const EVENT_SIZE = calculateEventSize();
const PRIMER_IDX = 355;  // shift 1s.
let lastSample = tf.scalar(PRIMER_IDX, 'int32');

const container = document.querySelector('#keyboard');
const keyboardInterface = new KeyboardElement(container);

const piano = new Piano({velocities: 4}).toMaster();

const SALAMANDER_URL = '/soundfonts/salamander-piano/';
const CHECKPOINT_URL = '/checkpoints/performance-rnn-tfjs';

const isDeviceSupported = tf.ENV.get('WEBGL_VERSION') >= 1;

if (!isDeviceSupported) {
    document.querySelector('#status').innerHTML =
        'We do not yet support your device. Please try on a desktop ' +
        'computer with Chrome/Firefox, or an Android phone with WebGL support.';
} else {
    start();
}

let modelReady = false;
let modelRunning = false;

let startButton = document.querySelector('#start-pause-button') as HTMLButtonElement;

function start() {
    piano.load(SALAMANDER_URL)
        .then(() => {
            return fetch(`${CHECKPOINT_URL}/weights_manifest.json`)
                .then((response) => response.json())
                .then(
                    (manifest: tf.WeightsManifestConfig) =>
                        tf.loadWeights(manifest, CHECKPOINT_URL));
        })
        .then((vars: { [varName: string]: tf.Tensor }) => {
            document.querySelector('#status').classList.add('hidden');
            document.querySelector('#controls').classList.remove('hidden');
            document.querySelector('#keyboard').classList.remove('hidden');

            lstmKernel1 =
                vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'] as
                    tf.Tensor2D;
            lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'] as
                tf.Tensor1D;

            lstmKernel2 =
                vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'] as
                    tf.Tensor2D;
            lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'] as
                tf.Tensor1D;

            lstmKernel3 =
                vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/kernel'] as
                    tf.Tensor2D;
            lstmBias3 = vars['rnn/multi_rnn_cell/cell_2/basic_lstm_cell/bias'] as
                tf.Tensor1D;

            fcB = vars['fully_connected/biases'] as tf.Tensor1D;
            fcW = vars['fully_connected/weights'] as tf.Tensor2D;
            modelReady = true;
            enableResumeButton();
        });
}

function resetRnn() {
    c = [
        tf.zeros([1, lstmBias1.shape[0] / 4]),
        tf.zeros([1, lstmBias2.shape[0] / 4]),
        tf.zeros([1, lstmBias3.shape[0] / 4]),
    ];
    h = [
        tf.zeros([1, lstmBias1.shape[0] / 4]),
        tf.zeros([1, lstmBias2.shape[0] / 4]),
        tf.zeros([1, lstmBias3.shape[0] / 4]),
    ];
    if (lastSample != null) {
        lastSample.dispose();
    }
    lastSample = tf.scalar(PRIMER_IDX, 'int32');
    currentPianoTimeSec = piano.now();
    currentLoopId++;
    generateStep(currentLoopId);
}

window.addEventListener('resize', resize);

function resize() {
    keyboardInterface.resize();
}

resize();

const densityControl =
    document.getElementById('note-density') as HTMLInputElement;
const densityDisplay = document.getElementById('note-density-display');

const gainSliderElement = document.getElementById('gain') as HTMLInputElement;
const gainDisplayElement =
    document.getElementById('gain-display') as HTMLSpanElement;
let globalGain = +gainSliderElement.value;
gainDisplayElement.innerText = globalGain.toString();
gainSliderElement.addEventListener('input', () => {
    globalGain = +gainSliderElement.value;
    gainDisplayElement.innerText = globalGain.toString();
});

function updateConditioningParams() {
    if (noteDensityEncoding != null) {
        noteDensityEncoding.dispose();
        noteDensityEncoding = null;
    }

    const noteDensityIdx = parseInt(densityControl.value, 10) || 0;
    const noteDensity = DENSITY_BIN_RANGES[noteDensityIdx];
    densityDisplay.innerHTML = noteDensity.toString();

    noteDensityEncoding =
        tf.oneHot(
            tf.tensor1d([noteDensityIdx + 1], 'int32'),
            DENSITY_BIN_RANGES.length + 1).as1D();
}

/**
 * Set the relative frequency of the notes generated by the model.
 *
 * @param values
 *  An array of 12 numbers, one for each note, representing the relative
 *  frequency of each note. The numbers do not need to sum to 1.
 */
function setPitchWeights(values: Array<number>) {
    if (PITCH_WEIGHT_SIZE != values.length) {
        throw new Error(`Wrong number of pitch weights (should be ${PITCH_WEIGHT_SIZE})`);
    }

    if (pitchDistribution != null) {
        pitchDistribution.dispose();
        pitchDistribution = null;
    }
    const buffer = tf.buffer<tf.Rank.R1>([PITCH_WEIGHT_SIZE], 'float32');
    const totalWeight = values.reduce((prev, val) => {
        return prev + val;
    });
    for (let i = 0; i < PITCH_WEIGHT_SIZE; i++) {
        buffer.set(values[i] / totalWeight, i);
    }
    pitchDistribution = buffer.toTensor();
}

setPitchWeights(DEFAULT_PITCH_WEIGHTS);
document.getElementById('note-density').oninput = updateConditioningParams;
updateConditioningParams();

document.getElementById('reset-rnn').onclick = () => {
    resetRnn();
};

function getConditioning(): tf.Tensor1D {
    return tf.tidy(() => {
        const axis = 0;
        const conditioningValues =
            noteDensityEncoding.concat(pitchDistribution, axis);
        return tf.tensor1d([0], 'int32').concat(conditioningValues, axis);
    });
}

async function generateStep(loopId: number) {
    if (loopId < currentLoopId) {
        // Was part of an outdated generateStep() scheduled via setTimeout.
        return;
    }

    const lstm1 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel1, lstmBias1, data, c, h);
    const lstm2 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel2, lstmBias2, data, c, h);
    const lstm3 = (data: tf.Tensor2D, c: tf.Tensor2D, h: tf.Tensor2D) =>
        tf.basicLSTMCell(forgetBias, lstmKernel3, lstmBias3, data, c, h);

    let outputs: tf.Scalar[] = [];
    [c, h, outputs] = tf.tidy(() => {
        // Generate some notes.
        const innerOuts: tf.Scalar[] = [];
        for (let i = 0; i < STEPS_PER_GENERATE_CALL; i++) {
            // Use last sampled output as the next input.
            const eventInput = tf.oneHot(
                lastSample.as1D(), EVENT_SIZE).as1D();
            // Dispose the last sample from the previous generate call, since we
            // kept it.
            if (i === 0) {
                lastSample.dispose();
            }
            const conditioning = getConditioning();
            const axis = 0;
            const input = conditioning.concat(eventInput, axis).toFloat();
            const output =
                tf.multiRNNCell([lstm1, lstm2, lstm3], input.as2D(1, -1), c, h);
            c.forEach(c => c.dispose());
            h.forEach(h => h.dispose());
            c = output[0];
            h = output[1];

            const outputH = h[2];
            const logits = outputH.matMul(fcW).add(fcB);

            const sampledOutput = tf.multinomial(logits.as1D(), 1).asScalar();

            innerOuts.push(sampledOutput);
            lastSample = sampledOutput;
        }
        return [c, h, innerOuts] as [tf.Tensor2D[], tf.Tensor2D[], tf.Scalar[]];
    });

    for (let i = 0; i < outputs.length; i++) {
        playOutput(outputs[i].dataSync()[0]);
    }

    if (piano.now() - currentPianoTimeSec > MAX_GENERATION_LAG_SECONDS) {
        console.warn(
            `Generation is ${piano.now() - currentPianoTimeSec} seconds behind, ` +
            `which is over ${MAX_NOTE_DURATION_SECONDS}. Resetting time!`);
        currentPianoTimeSec = piano.now();
    }
    const delta = Math.max(
        0, currentPianoTimeSec - piano.now() - GENERATION_BUFFER_SECONDS);
    stepTimeout = setTimeout(() => generateStep(loopId), delta * 1000);
}

/**
 * Decode the output index and play it on the piano and keyboardInterface.
 */
function playOutput(index: number) {
    let offset = 0;
    for (const eventRange of EVENT_RANGES) {
        const eventType = eventRange[0] as string;
        const minValue = eventRange[1] as number;
        const maxValue = eventRange[2] as number;
        if (offset <= index && index <= offset + maxValue - minValue) {
            if (eventType === 'note_on') {
                const noteNum = index - offset;
                setTimeout(() => {
                    keyboardInterface.keyDown(noteNum);
                    setTimeout(() => {
                        keyboardInterface.keyUp(noteNum);
                    }, 100);
                }, (currentPianoTimeSec - piano.now()) * 1000);
                activeNotes.set(noteNum, currentPianoTimeSec);

                return piano.keyDown(
                    noteNum, currentPianoTimeSec, currentVelocity * globalGain / 100);
            } else if (eventType === 'note_off') {
                const noteNum = index - offset;

                const activeNoteEndTimeSec = activeNotes.get(noteNum);
                // If the note off event is generated for a note that hasn't been
                // pressed, just ignore it.
                if (activeNoteEndTimeSec == null) {
                    return;
                }
                const timeSec =
                    Math.max(currentPianoTimeSec, activeNoteEndTimeSec + .5);

                piano.keyUp(noteNum, timeSec);
                activeNotes.delete(noteNum);
                return;
            } else if (eventType === 'time_shift') {
                currentPianoTimeSec += (index - offset + 1) / STEPS_PER_SECOND;
                activeNotes.forEach((timeSec, noteNum) => {
                    if (currentPianoTimeSec - timeSec > MAX_NOTE_DURATION_SECONDS) {
                        console.info(
                            `Note ${noteNum} has been active for ${
                                currentPianoTimeSec - timeSec}, ` +
                            `seconds which is over ${MAX_NOTE_DURATION_SECONDS}, will ` +
                            `release.`);

                        piano.keyUp(noteNum, currentPianoTimeSec);
                        activeNotes.delete(noteNum);
                    }
                });
                return currentPianoTimeSec;
            } else if (eventType === 'velocity_change') {
                currentVelocity = (index - offset + 1) * Math.ceil(127 / VELOCITY_BINS);
                currentVelocity = currentVelocity / 127;
                return currentVelocity;
            } else {
                throw new Error('Could not decode eventType: ' + eventType);
            }
        }
        offset += maxValue - minValue + 1;
    }
    throw new Error(`Could not decode index: ${index}`);
}

// Reset the RNN repeatedly so it doesn't trail off into incoherent musical
// babble.
const resettingText = document.getElementById('resettingText');

function resetRnnRepeatedly() {
    if (modelReady) {
        resetRnn();
        resettingText.style.opacity = '100';
    }

    setTimeout(() => {
        resettingText.style.opacity = '0';
    }, 1000);
    resetTimeout = setTimeout(resetRnnRepeatedly, RESET_RNN_FREQUENCY_MS);
}

function pauseModel() {
    if (stepTimeout != null) {
        clearTimeout(stepTimeout);
        stepTimeout = null;
    }
    if (resetTimeout != null) {
        clearTimeout(resetTimeout);
        resetTimeout = null;
    }
    modelRunning = false;
}

function startModel() {
    if (modelReady) {
        modelRunning = true;
        resetRnnRepeatedly();
    }
}

function enableResumeButton() {
    startButton.removeAttribute('disabled');
    startButton.classList.remove('disabled');
}

startButton.addEventListener('click', () => {
    if (modelRunning) {
        pauseModel();
        startButton.innerHTML = 'Play';
    } else {
        startModel();
        startButton.innerHTML = 'Pause';
    }
});
