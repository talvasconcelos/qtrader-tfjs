const rawData = require('./dataset')

const ohcl4 = (data) => {
    return data.reduce((a, b) => a + b, 0) / 4
}

const getData = () => {
    return rawData.map(cur => +cur.close)
}

// const getDiff = () => {
//     const last = data[]
// }

// let data = rawData.map(cur => ohcl4([cur.open, cur.high, cur.low, cur.close]))
// data = data.map((cur, i) => {
//     if(i >= 1){
//         let last = data[i - 1]
//         return (cur - last) / last
//     }
//     return 0
// })

// data = data.map((cur,i) => {
//     if(i >= 4){
//         return data.slice(i - 4, i + 1)
//     }
//     return null
// }).filter(v => v)


module.exports = getData


