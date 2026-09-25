import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { AlertsProvider } from './hooks/useAlerts'
import './styles.css'
import './components/alerts/alerts.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><AlertsProvider><App /></AlertsProvider></React.StrictMode>,
)
