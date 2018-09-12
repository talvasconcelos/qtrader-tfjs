class Agent{
    constructor(stateSize, is_eval = false, modelName = '') {
        this.stateSize = stateSize
        this.actionSize = 3
        this.memory = []
        this.inventory = []
        this.modelName = modelName
        this.eval = is_eval

        this.gamma = 0.95
        this.epsilon = 1.0
        this.epsilonMin = 0.01
        this.epsilonDecay = 0.995

        this.model = is_eval ? tf.loadModel('models/' + modelName) : this._model()
    }

    _model() {
        const model = tf.sequential()
        model.add(tf.layers.dense({
            units: 64,
            inputDim:  this.stateSize,
            activation: 'relu'
        }))
        model.add(tf.layers.dense({units: 32, activation: 'relu'}))
        model.add(tf.layers.dense({units: 8, activation: 'relu'}))
        model.add(tf.layers.dense({units: this.actionSize, activation: 'linear'}))
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
    
            state = tf.tensor(state).reshape([-1, this.stateSize])
            let options = this.model.predict(state).dataSync()
            // console.log(tf.argMax(options).dataSync()[0])
            return tf.argMax(options).dataSync()[0]
        })        
    }

    expReplay(batchSize) {
        return tf.tidy(() => {
            let miniBatch = []
            let l = this.memory.length
            let range = _.range(l - batchSize + 1, l)

            for (let i = 0; i < range.length; i++) {
                miniBatch.push(this.memory[i])
            }

            miniBatch.map(async cur => {
                let [state, action, reward, next_state, done] = cur
                
                state = tf.tensor(state).reshape([-1, this.stateSize])
                next_state = tf.tensor(next_state).reshape([-1, this.stateSize])

                let target = reward
                if (!done) {
                    let predictNext = this.model.predict(next_state)
                    // let y = predictNext.argMax().dataSync()[0]
                    target = reward + this.gamma * predictNext.argMax().dataSync()[0]
                }

                let target_f = this.model.predict(state).dataSync()
                target_f[action] = target
                await this.model.fit(state, target_f, {
                    epochs: 1
                })
            })

            if (this.epsilon > this.epsilonMin) {
                this.epsilon *= this.epsilonDecay
            }
        })
        
    }
}
