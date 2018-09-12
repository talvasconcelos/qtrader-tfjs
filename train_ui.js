const startTrain = () => {
    let [pair_name, window_size, episode_count] = ['BTC-USD', '5', '10']
    window_size = +window_size
    episode_count = +episode_count

    let agent = new Agent(window_size)
    let data = getData(pair_name)
    let l = data.length - 1
    let batch_size = 32


    for (let e = 0; e < episode_count + 1; e++) {
        console.log('Episode ' + e + '/' + episode_count)
        let state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []
        const div = document.createElement('div')
        const p = document.createElement('p')

        for (let t = 0; t < l; t++) {
            let action = agent.action(state)
            //sit
            let next_state = getState(data, t + 1, window_size + 1)
            let reward = 0

            if (action == 1 && agent.inventory.length === 0) { //buy
                agent.inventory.push(data[t])
                p.innerText = 'Buy: ' + data[t]
                div.appendChild(p)
                console.log('Buy: ' + data[t])
            } else if (action == 2 && agent.inventory.length > 0) { //sell
                let bought_price = agent.inventory.shift(0)
                let _profit = data[t] - bought_price
                let pct = (_profit / bought_price)
                reward = _profit <= 0 ? 0 : pct <= 0.1 ? 1 : 2 //_.max([_profit, 0])
                total_profit += _profit
                // console.log(reward)
                p.innerText = 'Sell: ' + data[t] + ' | Profit: ' + _profit
                div.appendChild(p)
                console.log('Sell: ' + data[t] + ' | Profit: ' + _profit)
            }

            let done = t == (l - 1)

            agent.memory.push([state, action, reward, next_state, done])
            state = next_state

            if (done) {
                console.log('--------------------------------')
                console.log('Total Profit:', total_profit)
                console.log('--------------------------------')
            }

            if (agent.memory.length > batch_size) {
                agent.expReplay(batch_size)
            }
        }

        // if(e % 10 == 10){
        //     agent.model.save('models/model_ep' + e)
        // }
    }
    console.log('Done')
}