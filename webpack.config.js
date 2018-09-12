const path = require('path')

module.exports = {
  node: {
    fs: 'empty'
  },
  entry: './index.js',
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist')
  },
  watch: true,
}