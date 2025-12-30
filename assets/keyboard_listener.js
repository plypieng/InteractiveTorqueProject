
document.addEventListener('keydown', function(event) {
    // Only trigger if not in an input or textarea
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        return;
    }

    const key = event.key.toLowerCase();
    const allowedKeys = ['p', 'f', 'enter'];

    if (allowedKeys.includes(key)) {
        // We need to trigger a Dash callback. 
        // A common trick is to use a hidden button or update a store if we can access it.
        // However, accessing Dash stores from outside React/Dash is tricky.
        // The most reliable way in pure Dash without extensions is to use dash_clientside.set_props
        // if available (Dash 2.11+), or trigger a click on a hidden button.
        
        // Let's try to find a hidden button for each action if we want direct action,
        // OR update a store.
        
        // For now, let's assume we will use dash_clientside.set_props if available.
        if (window.dash_clientside && window.dash_clientside.set_props) {
            window.dash_clientside.set_props("keyboard-event", {data: {key: key, ts: Date.now()}});
        } else {
            console.warn("dash_clientside.set_props not available. Keyboard shortcuts might not work.");
        }
    }
});
