const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

// Load the binding:
require('@tensorflow/tfjs-node')

class Agent{
    constructor(stateSize, is_eval = false, modelName) {
        this.stateSize = stateSize
        this.actionSize = 3
        this.memory = []
        this.inventory = []
        this.modelName = modelName
        this.eval = is_eval

        this.gamma = 0.95
        this.epsilon = 1.0
        this.epsilonMin = 0.01
        this.epsilonDecay = 0.95

        this.model = is_eval ? this.loadModel(modelName) : this._model()
    }

    async loadModel(model) {
        const loaded_model = await tf.loadModel('file://models/' + model)
        return loaded_model
    }

    _model() {
        const model = tf.sequential()
        model.add(tf.layers.dense({
            units: 32,
            inputShape: [this.stateSize],
            //inputDim:  this.stateSize,
            activation: 'elu'
        }))
        model.add(tf.layers.dense({units: 16, activation: 'elu'}))
        model.add(tf.layers.dense({units: 8, activation: 'elu'}))
        model.add(tf.layers.dense({units: this.actionSize, activation: 'softmax'}))
        model.compile({
            optimizer: tf.train.adadelta(),
            loss: 'meanSquaredError',
            metrics: ['accuracy']
        })

        return model
    }

    action(state) {
        return tf.tidy(() => {
            if(!this.eval && Math.random() <= this.epsilon){
                return _.random(this.actionSize)
            }

            let _state = tf.tensor(state).reshape([-1, this.stateSize])
            let options = this.model.predict(_state)
            //options.print()
            return tf.argMax(options).dataSync()[0]
        })
    }

    async expReplay(batchSize) {

        let X = []
        let Y = []
                
        await this.memory.map((mem, i) => {
            
            let [state, action, reward, next_state, done] = mem
            let target = reward
            
            let _state = tf.tensor(state).reshape([-1, this.stateSize])
            let _next_state = tf.tensor(next_state).reshape([-1, this.stateSize])

            if (!done) {
                let predictNext = this.model.predict(_next_state)
                target = reward + this.gamma * predictNext.argMax().dataSync()[0]
                //console.log(predictNext.dataSync())
            }

            let target_f = this.model.predict(_state).dataSync()
            target_f[action] = target

            _state.dispose()
            _next_state.dispose()

            X.push(state)
            Y.push(Array.from(target_f))
        })
        //console.log(X.slice(0, 11))
        X = tf.tensor(X).reshape([-1, this.stateSize])
        Y = tf.tensor2d(Y)
        
        // X.print()
        // this.model.predict(X).print()

        await this.model.fit(X, Y, {
            epochs: 1,
            batchSize,
            validationSplit: 0.3,
            shuffle: true,
            callbacks: {
                // onBatchEnd: (batch, logs) => {
                //     console.log(`${logs.loss} ${logs.acc}`)
                // },
                onEpochEnd: (epoch, logs) => {
                    console.log(`${logs.val_loss} ${logs.val_acc}`)
                }
            }
        })
        

        if (this.epsilon > this.epsilonMin) {
            this.epsilon *= this.epsilonDecay
        }

        X.dispose()
        Y.dispose()

    }
}

module.exports = Agent
