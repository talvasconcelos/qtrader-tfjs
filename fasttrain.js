const args = process.argv
const _ = require('lodash')
const {getData, getState} = require('./utils')

const Agent = require('./agent/fastagent')
console.log('Start script with: node --max-old-space-size=4096')

if(args.length !== 5){
    console.log('Usage: node train [PAIR] [WINDOW] [EPISODES]')
    process.exit(-1)
}

let [pair_name, window_size, episode_count] = args.slice(2)
window_size = +window_size
episode_count = +episode_count

let agent = new Agent(window_size)
let data = getData(pair_name)
let l = data.length - 1
let batch_size = 32
const MAX_MEM = 5000
const buy_hold = (data[l] - data[0])

const train = async () => {

    for (let e = 0; e < episode_count + 1; e++) {
        console.log('Episode ' + e + '/' + episode_count)
        let state = getState(data, 0, window_size + 1)
        
        total_profit = 0
        agent.inventory = []
        total_trades = 0

        for (let t = 0; t < l; t++) {
            let action = agent.action(state)
            
            //console.log(action)
            //sit
            let next_state = getState(data, t + 1, window_size + 1)
            let reward = 0.25

            if(action == 1 && agent.inventory.length === 0) { //buy
                reward += 0.25
                agent.inventory.push(data[t])
                // console.log(`Buy: ${data[t]} | Ep: ${e} | D: ${t}`)
            } else if(action === 2 && agent.inventory.length > 0) { //sell
                let bought_price = agent.inventory.shift(0)
                let _profit = data[t] - bought_price
                let pct = (_profit / bought_price)
                reward = _profit <= 0 ? -1 : pct <= 0.1 ? 0.5 : 1 + pct//_.max([_profit, 0])
                total_profit += isNaN(_profit) ? 0 : _profit
                reward += total_profit > buy_hold ? 1 : -1
                total_trades++
                // console.log(reward)
                // console.log(`Sell: ${data[t]} | Profit: ${ _profit}`)
            }

            let done = t === (l - 1)

            if(agent.memory.length > MAX_MEM - 1) {
                agent.memory.shift()
            }
            agent.memory.push([state, action, reward, next_state, done])
            // console.log(agent.memory.length)
            state = next_state

            if(done) {
                console.log('--------------------------------')
                console.log('Total Profit:', total_profit.toFixed(2), 'Trades:', total_trades)
                console.log('Vs Buy/Hold:', (total_profit - buy_hold).toFixed(2))
                console.log('--------------------------------')
            }

        }
        await agent.expReplay(batch_size)
        if(e % 10 === 0 && e !== 0){
            await agent.model.save('file://models/modelHP-ep' + e)
        }
    }
    console.log('Done')
}

train()
