const args = process.argv
const _ = require('lodash')
const {getData, getState} = require('./utils')

const Agent = require('./agent/fastagent')
console.log('Start script with: node --max-old-space-size=4096')

if(args.length !== 4){
    console.log('Usage: node train [PAIR] [MODEL]')
    process.exit(-1)
}    

let [pair_name, modelName] = args.slice(2)

// const model = tf.loadModel('')
// const window_size = model.layers

let agent = new Agent(window_size, true, modelName)
let data = getData(pair_name)
let l = data.length - 1
let batch_size = 32

let state = getState(data, 0, window_size + 1)
let total_profit = 0
agent.inventory = []
   

for (let t = 0; t < l; t++) {
    let action = agent.action(state)

    //sit
    let next_state = getState(data, t + 1, window_size + 1)
    let reward = 0

    if(action == 1 && agent.inventory.length === 0) { //buy
        agent.inventory.push(data[t])
        // console.log('Buy: ' + data[t])
    } else if(action == 2 && agent.inventory.length > 0) { //sell
        let bought_price = agent.inventory.shift(0)
        let _profit = data[t] - bought_price
        let pct = (_profit / bought_price)
        reward = _profit <= 0 ? 0 : pct <= 0.1 ? pct * 10 : pct * 100//_.max([_profit, 0])
        total_profit += _profit
        // console.log(reward)
        // console.log('Sell: ' + data[t] + ' | Profit: ' + _profit)
    }

    let done = t == (l - 1)

    if(agent.memory.length > MAX_MEM) {
        agent.memory.shift()
    }        
    agent.memory.push([state, action, reward, next_state, done])
    // console.log(agent.memory.length)
    state = next_state

    if(done) {
        console.log('--------------------------------')
        console.log('Total Profit:', total_profit)
        console.log('--------------------------------')
    }
}

const evaluate = async (eval_pair) => {
    agent.eval = true
    data = getData(eval_pair)

    state = getState(data, 0, window_size + 1)
        
    total_profit = 0
    agent.inventory = []
    total_trades = 0

    for (let t = 0; t < data.length - 1; t++) {
        let action = agent.action(state)
        
        //console.log(action)
        //sit
        let next_state = getState(data, t + 1, window_size + 1)
        let reward = 0

        if(action == 1 && agent.inventory.length === 0) { //buy
            agent.inventory.push(data[t])
            console.log(`Buy: ${data[t]} | Ep: ${e} | D: ${t}`)
        } else if(action == 2 && agent.inventory.length > 0) { //sell
            let bought_price = agent.inventory.shift(0)
            let _profit = data[t] - bought_price
            let pct = (_profit / bought_price)
            reward = _profit <= 0 ? 0.01 : pct <= 0.1 ? pct : pct + 1//_.max([_profit, 0])
            total_profit += isNaN(_profit) ? 0 : _profit
            total_trades++
            // console.log(reward)
            console.log(`Sell: ${data[t]} | Profit: ${ _profit}`)
        }

        let done = t == (l - 1)

        // if(agent.memory.length > MAX_MEM - 1) {
        //     agent.memory.shift()
        // }
        // agent.memory.push([state, action, reward, next_state, done])
        // console.log(agent.memory.length)
        state = next_state

        if(done) {
            console.log('--------------------------------')
            console.log('Total Profit:', total_profit.toFixed(2), 'Trades:', total_trades)
            console.log('--------------------------------')
        }

    }
}

console.log('Evaluation Done')