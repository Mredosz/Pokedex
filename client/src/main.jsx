import {StrictMode} from 'react'
import {createRoot} from 'react-dom/client'
import './index.css';
import MainPage from "./pages/MainPage.jsx";
import {QueryClient, QueryClientProvider} from "@tanstack/react-query";

const queryClient = new QueryClient();

createRoot(document.getElementById('root')).render(
    <StrictMode>
        <QueryClientProvider client={queryClient}>
            <MainPage/>
        </QueryClientProvider>
    </StrictMode>,
)
