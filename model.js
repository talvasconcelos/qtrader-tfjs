const tf = require('@tensorflow/tfjs')

//require('@tensorflow/tfjs-node')

const model = tf.sequential()

model.add(tf.layers.lstm({
    inputShape: [5, 1],
    units: 5,
    returnSequences: true
}))

model.add(tf.layers.lstm({
    units: 24,
    returnSequences: false
}))

model.add(tf.layers.dense({units: 5}))

model.compile({
    optimizer: tf.train.adadelta(),
    loss: 'meanSquaredError',
    metrics: ['accuracy']
})

model.summary()

const n = [0, 1, 2, 8, 10, 25, 0, 98, 100, 1500]

for (let i = 0; i < n.length; i++) {
    console.log(1 / (1 + Math.exp(-n[i])))
    
}