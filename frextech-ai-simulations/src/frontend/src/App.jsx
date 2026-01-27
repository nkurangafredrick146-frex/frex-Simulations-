import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate
} from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import {
  Snackbar,
  Alert,
  LinearProgress,
  Backdrop,
  CircularProgress
} from '@mui/material';

// Context Providers
import { AuthProvider } from './contexts/AuthContext';
import { WorldProvider } from './contexts/WorldContext';
import { EditorProvider } from './contexts/EditorContext';
import { NotificationProvider } from './contexts/NotificationContext';

// Components
import Layout from './components/layout/Layout';
import ProtectedRoute from './components/auth/ProtectedRoute';

// Pages
import Dashboard from './pages/Dashboard';
import WorldCreator from './pages/WorldCreator';
import WorldEditor from './pages/WorldEditor';
import WorldExplorer from './pages/WorldExplorer';
import AssetLibrary from './pages/AssetLibrary';
import Settings from './pages/Settings';
import Login from './pages/Login';
import Signup from './pages/Signup';
import Documentation from './pages/Documentation';

// Services
import ApiService from './services/ApiService';
import WebSocketService from './services/WebSocketService';
import ErrorBoundary from './components/error/ErrorBoundary';

// Theme Configuration
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00d4ff',
      light: '#66e3ff',
      dark: '#00a3cc'
    },
    secondary: {
      main: '#9d4edd',
      light: '#b87fed',
      dark: '#7c2eb3'
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a2e'
    },
    error: {
      main: '#ff4757'
    },
    success: {
      main: '#2ed573'
    },
    warning: {
      main: '#ffa502'
    }
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      fontSize: '3.5rem',
      background: 'linear-gradient(135deg, #00d4ff 0%, #9d4edd 100%)',
      WebkitBackgroundClip: 'text',
      WebkitTextFillColor: 'transparent'
    },
    h2: {
      fontWeight: 600,
      fontSize: '2.5rem'
    },
    h3: {
      fontWeight: 600,
      fontSize: '2rem'
    },
    h4: {
      fontWeight: 500,
      fontSize: '1.5rem'
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.6
    },
    button: {
      fontWeight: 600,
      textTransform: 'none'
    }
  },
  shape: {
    borderRadius: 12
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: '10px 24px'
        }
      }
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }
      }
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          background: 'rgba(10, 10, 10, 0.95)',
          backdropFilter: 'blur(20px)',
          borderRight: '1px solid rgba(255, 255, 255, 0.1)'
        }
      }
    }
  }
});

/**
 * Main Application Component
 */
function App() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'info'
  });
  
  const apiService = useRef(new ApiService());
  const wsService = useRef(new WebSocketService());

  // Initialize services
  useEffect(() => {
    const initializeServices = async () => {
      try {
        setLoading(true);
        
        // Initialize API service
        await apiService.current.initialize();
        
        // Initialize WebSocket connection
        wsService.current.onConnect(() => {
          setConnectionStatus('connected');
          showNotification('Connected to simulation server', 'success');
        });
        
        wsService.current.onDisconnect(() => {
          setConnectionStatus('disconnected');
          showNotification('Disconnected from server', 'warning');
        });
        
        wsService.current.onError((error) => {
          console.error('WebSocket error:', error);
          showNotification('Connection error', 'error');
        });
        
        await wsService.current.connect();
        
        // Check for updates
        await checkForUpdates();
        
        setLoading(false);
        
      } catch (err) {
        console.error('Failed to initialize services:', err);
        setError(err.message);
        setLoading(false);
      }
    };
    
    initializeServices();
    
    // Cleanup
    return () => {
      wsService.current.disconnect();
    };
  }, []);
  
  // Check for application updates
  const checkForUpdates = useCallback(async () => {
    try {
      const updateInfo = await apiService.current.get('/api/version');
      const currentVersion = process.env.REACT_APP_VERSION;
      
      if (updateInfo.latestVersion !== currentVersion) {
        showNotification(
          `Update available: ${updateInfo.latestVersion}`,
          'info'
        );
      }
    } catch (error) {
      console.warn('Failed to check for updates:', error);
    }
  }, []);
  
  // Show notification
  const showNotification = useCallback((message, severity = 'info') => {
    setNotification({
      open: true,
      message,
      severity
    });
  }, []);
  
  // Close notification
  const handleCloseNotification = useCallback(() => {
    setNotification(prev => ({ ...prev, open: false }));
  }, []);
  
  // Handle global errors
  const handleGlobalError = useCallback((error, errorInfo) => {
    console.error('Global error:', error, errorInfo);
    setError(error.message);
    
    // Send error to monitoring service
    if (apiService.current) {
      apiService.current.post('/api/monitoring/errors', {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo?.componentStack,
        timestamp: new Date().toISOString()
      }).catch(console.error);
    }
  }, []);
  
  // Retry connection
  const retryConnection = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      await wsService.current.connect();
      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }, []);
  
  if (loading) {
    return (
      <Backdrop open={true} sx={{ color: '#fff', zIndex: 9999 }}>
        <CircularProgress color="primary" />
      </Backdrop>
    );
  }
  
  if (error) {
    return (
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        padding: '20px',
        textAlign: 'center'
      }}>
        <h1 style={{ color: '#ff4757', marginBottom: '20px' }}>
          Application Error
        </h1>
        <p style={{ marginBottom: '30px', maxWidth: '600px' }}>
          {error}
        </p>
        <button
          onClick={retryConnection}
          style={{
            background: 'linear-gradient(135deg, #00d4ff 0%, #9d4edd 100%)',
            color: 'white',
            border: 'none',
            padding: '12px 32px',
            borderRadius: '8px',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            transition: 'transform 0.2s'
          }}
        >
          Retry Connection
        </button>
      </div>
    );
  }
  
  return (
    <ErrorBoundary onError={handleGlobalError}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        
        <NotificationProvider value={{ showNotification }}>
          <AuthProvider apiService={apiService.current}>
            <WorldProvider apiService={apiService.current} wsService={wsService.current}>
              <EditorProvider>
                <Router>
                  <Layout connectionStatus={connectionStatus}>
                    <Routes>
                      {/* Public Routes */}
                      <Route path="/login" element={<Login />} />
                      <Route path="/signup" element={<Signup />} />
                      <Route path="/docs" element={<Documentation />} />
                      
                      {/* Protected Routes */}
                      <Route path="/" element={
                        <ProtectedRoute>
                          <Dashboard />
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/create" element={
                        <ProtectedRoute>
                          <WorldCreator />
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/editor/:worldId" element={
                        <ProtectedRoute>
                          <WorldEditor />
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/explore/:worldId" element={
                        <ProtectedRoute>
                          <WorldExplorer />
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/assets" element={
                        <ProtectedRoute>
                          <AssetLibrary />
                        </ProtectedRoute>
                      } />
                      
                      <Route path="/settings" element={
                        <ProtectedRoute>
                          <Settings />
                        </ProtectedRoute>
                      } />
                      
                      {/* Redirect unknown routes */}
                      <Route path="*" element={<Navigate to="/" replace />} />
                    </Routes>
                  </Layout>
                </Router>
              </EditorProvider>
            </WorldProvider>
          </AuthProvider>
        </NotificationProvider>
        
        {/* Global Notification Snackbar */}
        <Snackbar
          open={notification.open}
          autoHideDuration={6000}
          onClose={handleCloseNotification}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={handleCloseNotification}
            severity={notification.severity}
            elevation={6}
            variant="filled"
          >
            {notification.message}
          </Alert>
        </Snackbar>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

// Export App as default
export default App;

// Initialize React app
const rootElement = document.getElementById('root');
if (rootElement) {
  const root = ReactDOM.createRoot(rootElement);
  root.render(<App />);
}

// Export additional utilities for debugging
export { theme };