import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

// Role hierarchy - defines what each role can access
const ROLE_PERMISSIONS = {
  superuser: ['dashboard', 'knowledge-base', 'add-document', 'add-text', 'query-analytics', 'unsolved-queries', 'bot-settings', 'system-instructions', 'archive', 'user-settings'],
  admin: ['dashboard', 'knowledge-base', 'add-document', 'add-text', 'query-analytics', 'unsolved-queries', 'bot-settings', 'system-instructions', 'archive', 'user-settings'],
  assistant: ['dashboard', 'knowledge-base', 'query-analytics', 'unsolved-queries', 'bot-settings', 'archive', 'user-settings'],
};

export function RoleProtectedRoute({ children, requiredRoles = [], routeName = '' }) {
  const { user, isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-neutral-50">
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 animate-glow mb-4">
            <svg className="animate-spin h-8 w-8 text-white" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
          <p className="text-neutral-600 font-medium">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  const userRole = user?.role || 'assistant';
  
  // Check if user's role has access to this route
  if (requiredRoles.length > 0 && !requiredRoles.includes(userRole)) {
    return <Navigate to="/unauthorized" replace />;
  }

  // Alternative check using route name and role permissions
  if (routeName && ROLE_PERMISSIONS[userRole] && !ROLE_PERMISSIONS[userRole].includes(routeName)) {
    return <Navigate to="/unauthorized" replace />;
  }

  return children;
}

export default RoleProtectedRoute;
