const statusElement = document.getElementById('status')
const messageElement = document.getElementById('message')

export const isTraining = () => {
    statusElement.innerText = 'Training...'
}