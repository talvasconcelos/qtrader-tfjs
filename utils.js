const rawData = require('./dataset')
const _ = require('lodash')

const getData = () => {
    return rawData.map(cur => +cur.close)
}

const sigmoid = (x) => (1 / (1 + Math.exp(-x)))
const tanh = (x) => ((Math.tanh(x) + 1) / 2) 

const getState = (data, t, n) => {
    let d = t - n + 1
    let block
    if(d >= 0){
        block = data.slice(d, t + 1)
    } else {
        //let x = Math.abs(d)
        let pad = _.fill(Array(Math.abs(d)), 0)
        let _data = data.slice(0, t + 1)
        block = [...pad, ..._data]
    }
    let res = []
    for (let i = 0; i < n - 1; i++) {
        //let output = (block[i + 1] - block[i]) / block[i + 1]
        res.push(tanh(block[i + 1] - block[i]))
        //res.push(output)
    }
    
    return res
}

module.exports = {
    getData,
    getState
}