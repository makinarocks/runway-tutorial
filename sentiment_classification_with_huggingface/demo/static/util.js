function showMessage(message, type) {
	document.querySelector('.message').className = `message ${type}`;
	document.querySelector('.message').innerHTML = message;
}

function hideMessage() {
	document.querySelector('.message').innerHTML = '';
}

function showLoading() {
	document.querySelector('.result').classList.add('loading');
};

function hideLoading() {
	document.querySelector('.result').classList.remove('loading');
}
