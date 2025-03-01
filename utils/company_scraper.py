"""
Company Website Scraper

This module provides utilities to scrape content from a company website,
including multiple pages, to provide richer context for personality analysis.
"""

import requests
import html2text
import logging
import urllib3
import asyncio
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from typing import List, Dict, Set, Tuple, Optional
import time

# Suppress InsecureRequestWarning messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Use the centralized logger
logger = logging.getLogger(__name__)

# Initialize HTML to text converter
h2t = html2text.HTML2Text()
h2t.ignore_links = False  # Keep links to find more pages
h2t.ignore_images = True
h2t.ignore_tables = False
h2t.ignore_emphasis = True
h2t.body_width = 0  # No wrapping

# Update configuration to preserve more content and structure
h2t.ignore_images = False  # Keep image references
h2t.images_to_alt = True   # Convert images to their alt text
h2t.protect_links = True   # Don't replace links with just their text
h2t.single_line_break = False  # Use multiple line breaks to preserve structure
h2t.unicode_snob = True    # Use unicode characters instead of ASCII
h2t.inline_links = True    # Keep links inline
h2t.ignore_anchors = False  # Keep anchor links
h2t.mark_code = True       # Preserve code blocks
h2t.pad_tables = True      # Improve table formatting
h2t.escape_snob = True     # Don't escape special characters
h2t.emphasis_mark = '*'    # Use * for emphasis

class CompanyScraper:
    """Scrapes content from a company website to enhance company context."""
    
    def __init__(self, base_url: str, max_pages: int = 10, timeout: int = 10, max_retries: int = 2):
        """
        Initialize the company scraper.
        
        Args:
            base_url: The main URL of the company website
            max_pages: Maximum number of pages to scrape
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.base_url = self._normalize_url(base_url)
        self.max_pages = max_pages
        self.timeout = timeout
        self.max_retries = max_retries
        self.visited_urls: Set[str] = set()
        self.pages_content: Dict[str, str] = {}
        self.important_pages: List[str] = []
        
        # Parse the domain for later use
        parsed_url = urlparse(self.base_url)
        self.domain = parsed_url.netloc
        
    def _normalize_url(self, url: str) -> str:
        """Ensure URL is properly formatted with protocol."""
        if not url:
            return ""
            
        url = url.strip()
        if not url:
            return ""
            
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = f"https://{url}"
            
        # Remove trailing slash if present
        if url.endswith('/'):
            url = url[:-1]
            
        return url
    
    def _is_same_domain(self, url: str) -> bool:
        """Check if URL belongs to the same domain as the base URL."""
        try:
            parsed_url = urlparse(url)
            return parsed_url.netloc == self.domain
        except:
            return False
    
    def _extract_links(self, html_content: str, current_url: str) -> List[str]:
        """Extract all links from HTML content that belong to the same domain."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            # Find all link elements and extract href attributes
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Handle relative URLs
                if href.startswith('/'):
                    full_url = urljoin(self.base_url, href)
                    links.append(full_url)
                # Skip fragment identifiers, javascript, mailto, tel links
                elif href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:') or href.startswith('tel:'):
                    continue
                # Add complete URLs if they belong to the same domain
                else:
                    try:
                        # Check if it's a valid URL and belongs to the same domain
                        parsed_url = urlparse(href)
                        if parsed_url.netloc and parsed_url.netloc == self.domain:
                            links.append(href)
                        elif not parsed_url.netloc:
                            # It's a relative URL without a leading slash
                            full_url = urljoin(current_url, href)
                            if self._is_same_domain(full_url):
                                links.append(full_url)
                    except:
                        # Skip invalid URLs
                        continue
            
            # Remove duplicates and sort
            return sorted(list(set(links)))
        except Exception as e:
            logger.error(f"Error extracting links from {current_url}: {str(e)}")
            return []
    
    def _is_important_page(self, url: str) -> bool:
        """Determine if a page is important based on its URL pattern."""
        important_patterns = [
            r'/about', r'/company', r'/team', r'/mission', r'/vision',
            r'/services', r'/products', r'/solutions', r'/features',
            r'/contact', r'/careers', r'/jobs', r'/work',
            r'/customers', r'/clients', r'/case-studies', r'/testimonials',
            r'/pricing', r'/plans', r'/packages',
            r'/faq', r'/support', r'/help'
        ]
        
        # Check if URL matches any important pattern
        for pattern in important_patterns:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        
        # Check if it's the homepage
        parsed_url = urlparse(url)
        path = parsed_url.path
        if path == "" or path == "/" or path.lower() == "/index.html":
            return True
            
        return False
    
    async def _fetch_page(self, url: str) -> Tuple[str, str, List[str]]:
        """Fetch a single page and extract its content and links."""
        if url in self.visited_urls:
            return url, "", []
            
        self.visited_urls.add(url)
        
        for attempt in range(self.max_retries + 1):
            try:
                # Set user agent to avoid being blocked
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                # Make the request - using verify=False but warnings are suppressed
                response = requests.get(url, headers=headers, timeout=self.timeout, verify=False)
                response.raise_for_status()
                
                # Get the HTML content
                html_content = response.text
                
                # Extract metadata
                metadata = self._extract_metadata(html_content)
                
                # Pre-process HTML to improve extraction quality
                html_content = self._preprocess_html(html_content)
                
                # Extract links for further crawling
                links = self._extract_links(html_content, url)
                
                # Extract important sections
                important_sections = self._extract_important_sections(html_content)
                
                # Convert HTML to text
                text_content = h2t.handle(html_content)
                
                # Clean up the text while preserving structure
                text_content = self._clean_text_content(text_content)
                
                # Add metadata and important sections to the content
                if metadata:
                    metadata_text = "PAGE METADATA:\n" + "\n".join([f"{k}: {v}" for k, v in metadata.items()]) + "\n\n"
                    text_content = metadata_text + text_content
                
                if important_sections:
                    sections_text = "IMPORTANT SECTIONS:\n" + "\n\n".join([f"{k}:\n{v}" for k, v in important_sections.items()]) + "\n\n"
                    text_content = sections_text + text_content
                
                # Store the content
                self.pages_content[url] = text_content
                
                # Check if it's an important page
                if self._is_important_page(url):
                    self.important_pages.append(url)
                
                return url, text_content, links
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    # Wait before retrying
                    await asyncio.sleep(1)
                else:
                    logger.error(f"Error fetching {url}: {str(e)}")
                    return url, f"Error: {str(e)}", []
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                return url, f"Error: {str(e)}", []
    
    def _preprocess_html(self, html_content: str) -> str:
        """Pre-process HTML to improve extraction quality."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(["script", "style", "iframe"]):
                script_or_style.decompose()
            
            # Convert headings to be more prominent
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    # Add extra markers to make headings stand out
                    heading_text = heading.get_text()
                    if heading_text:
                        new_tag = soup.new_tag('div')
                        new_tag.string = f"{'#' * i} {heading_text} {'#' * i}"
                        heading.replace_with(new_tag)
            
            # Add extra spacing around important elements
            for elem in soup.find_all(['p', 'div', 'section', 'article']):
                if elem.get_text().strip():
                    elem['style'] = 'margin: 1em 0;'
            
            # Add markers to list items
            for li in soup.find_all('li'):
                li_text = li.get_text().strip()
                if li_text:
                    new_text = f"• {li_text}"
                    li.string = new_text
            
            # Improve table visualization
            for table in soup.find_all('table'):
                # Add a marker before each table
                marker = soup.new_tag('div')
                marker.string = "--- TABLE START ---"
                table.insert_before(marker)
                
                # Add a marker after each table
                marker = soup.new_tag('div')
                marker.string = "--- TABLE END ---"
                table.insert_after(marker)
            
            return str(soup)
        except Exception as e:
            logger.error(f"Error preprocessing HTML: {str(e)}")
            return html_content
    
    def _clean_text_content(self, text_content: str) -> str:
        """Clean up the text content while preserving structure."""
        # Remove excessive blank lines (more than 3) but preserve paragraph breaks
        text_content = re.sub(r'\n{4,}', '\n\n\n', text_content)
        
        # Normalize bullet points
        text_content = re.sub(r'^[*+-]\s+', '• ', text_content, flags=re.MULTILINE)
        
        # Preserve heading formatting
        text_content = re.sub(r'^(#+)\s*(.*?)\s*\1$', r'\1 \2', text_content, flags=re.MULTILINE)
        
        # Remove any remaining HTML comments
        text_content = re.sub(r'<!--.*?-->', '', text_content, flags=re.DOTALL)
        
        # Clean up any messy URL formatting
        text_content = re.sub(r'\]\(\s*(http[^)]+)\s*\)', r'](\1)', text_content)
        
        # Make sure there's a line with actual content
        lines = [line for line in text_content.splitlines() if line.strip()]
        if not lines:
            return "No meaningful content found on this page."
        
        # Join the lines back together
        return '\n'.join(lines)
    
    async def scrape_website(self, progress_callback=None) -> Dict[str, str]:
        """
        Scrape the company website to extract content from multiple pages.
        
        Args:
            progress_callback: Optional callback function to report progress
            
        Returns:
            Dictionary mapping URLs to their text content
        """
        # Reset state
        self.visited_urls = set()
        self.pages_content = {}
        self.important_pages = []
        
        # Start with the base URL
        urls_to_visit = [self.base_url]
        
        # Track progress
        total_pages_scraped = 0
        
        while urls_to_visit and total_pages_scraped < self.max_pages:
            # Get the next URL to visit
            current_url = urls_to_visit.pop(0)
            
            # Fetch the page
            _, _, new_links = await self._fetch_page(current_url)
            
            # Update progress
            total_pages_scraped += 1
            if progress_callback:
                progress_callback(total_pages_scraped, self.max_pages)
            
            # Add new links to the queue
            for link in new_links:
                if link not in self.visited_urls and link not in urls_to_visit:
                    urls_to_visit.append(link)
                    
                    # Prioritize important pages
                    if self._is_important_page(link):
                        # Move to the front of the queue
                        urls_to_visit.remove(link)
                        urls_to_visit.insert(0, link)
            
            # Small delay to avoid overloading the server
            await asyncio.sleep(0.5)
        
        return self.pages_content
    
    def get_combined_content(self, max_chars_per_page: int = 5000) -> str:
        """
        Get combined content from all scraped pages, prioritizing important pages.
        
        Args:
            max_chars_per_page: Maximum number of characters to include from each page
            
        Returns:
            Combined text content from all pages
        """
        combined_content = []
        
        # First add content from important pages
        for url in self.important_pages:
            if url in self.pages_content:
                page_content = self.pages_content[url]
                page_title = f"Page: {url}"
                combined_content.append(f"{page_title}\n{'-' * len(page_title)}\n{page_content[:max_chars_per_page]}\n\n")
        
        # Then add content from other pages
        for url, content in self.pages_content.items():
            if url not in self.important_pages:
                page_title = f"Page: {url}"
                combined_content.append(f"{page_title}\n{'-' * len(page_title)}\n{content[:max_chars_per_page]}\n\n")
        
        return "\n".join(combined_content)
    
    def get_important_pages_content(self, max_chars_per_page: int = 5000) -> str:
        """
        Get content only from important pages.
        
        Args:
            max_chars_per_page: Maximum number of characters to include from each page
            
        Returns:
            Combined text content from important pages
        """
        combined_content = []
        
        for url in self.important_pages:
            if url in self.pages_content:
                page_content = self.pages_content[url]
                page_title = f"Page: {url}"
                combined_content.append(f"{page_title}\n{'-' * len(page_title)}\n{page_content[:max_chars_per_page]}\n\n")
        
        return "\n".join(combined_content)

    def _extract_metadata(self, html_content: str) -> Dict[str, str]:
        """Extract metadata from HTML content."""
        metadata = {}
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                metadata['title'] = title_tag.string.strip()
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                metadata['description'] = meta_desc['content'].strip()
            
            # Extract meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords and meta_keywords.get('content'):
                metadata['keywords'] = meta_keywords['content'].strip()
            
            # Extract Open Graph metadata
            og_title = soup.find('meta', property='og:title')
            if og_title and og_title.get('content'):
                metadata['og:title'] = og_title['content'].strip()
                
            og_desc = soup.find('meta', property='og:description')
            if og_desc and og_desc.get('content'):
                metadata['og:description'] = og_desc['content'].strip()
            
            # Extract Twitter card metadata
            twitter_title = soup.find('meta', attrs={'name': 'twitter:title'})
            if twitter_title and twitter_title.get('content'):
                metadata['twitter:title'] = twitter_title['content'].strip()
                
            twitter_desc = soup.find('meta', attrs={'name': 'twitter:description'})
            if twitter_desc and twitter_desc.get('content'):
                metadata['twitter:description'] = twitter_desc['content'].strip()
                
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return metadata
    
    def _extract_important_sections(self, html_content: str) -> Dict[str, str]:
        """Extract important sections from the HTML content."""
        sections = {}
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract potential "About Us" sections
            about_selectors = [
                '#about', '.about', 'section.about', 'div.about',
                '[id*="about"]', '[class*="about"]',
                '#company', '.company', 'section.company', 'div.company'
            ]
            
            for selector in about_selectors:
                about_elements = soup.select(selector)
                for element in about_elements:
                    text = element.get_text().strip()
                    if text and len(text) > 100:  # Only include substantial text
                        sections['About'] = text[:1000] + "..." if len(text) > 1000 else text
                        break
                if 'About' in sections:
                    break
            
            # Extract potential "Services/Products" sections
            services_selectors = [
                '#products', '.products', 'section.products', 'div.products',
                '#services', '.services', 'section.services', 'div.services',
                '[id*="product"]', '[class*="product"]',
                '[id*="service"]', '[class*="service"]'
            ]
            
            for selector in services_selectors:
                service_elements = soup.select(selector)
                for element in service_elements:
                    text = element.get_text().strip()
                    if text and len(text) > 100:
                        sections['Products/Services'] = text[:1000] + "..." if len(text) > 1000 else text
                        break
                if 'Products/Services' in sections:
                    break
            
            # Extract potential "Contact" information
            contact_selectors = [
                '#contact', '.contact', 'section.contact', 'div.contact',
                '[id*="contact"]', '[class*="contact"]'
            ]
            
            for selector in contact_selectors:
                contact_elements = soup.select(selector)
                for element in contact_elements:
                    text = element.get_text().strip()
                    if text and len(text) > 50:
                        sections['Contact'] = text[:500] + "..." if len(text) > 500 else text
                        break
                if 'Contact' in sections:
                    break
            
            # Extract potential "Team" information
            team_selectors = [
                '#team', '.team', 'section.team', 'div.team',
                '[id*="team"]', '[class*="team"]'
            ]
            
            for selector in team_selectors:
                team_elements = soup.select(selector)
                for element in team_elements:
                    text = element.get_text().strip()
                    if text and len(text) > 100:
                        sections['Team'] = text[:1000] + "..." if len(text) > 1000 else text
                        break
                if 'Team' in sections:
                    break
                    
            return sections
        except Exception as e:
            logger.error(f"Error extracting important sections: {str(e)}")
            return sections

@st.dialog("Company Website Scraper", width="large")
def company_scraper_dialog(initial_url: str = ""):
    """
    Streamlit dialog for scraping a company website.
    
    Args:
        initial_url: Optional initial URL to populate the input field
    """
    st.write("Scrape your company website to enhance the company context.")
    
    # Get company URL - auto-populate if provided
    company_url = st.text_input(
        "Company Website URL",
        value=initial_url,
        placeholder="https://example.com",
        help="Enter your company's website URL to scrape content"
    )
    
    # Configuration options
    col1, col2 = st.columns(2)
    
    with col1:
        max_pages = st.slider(
            "Maximum Pages to Scrape",
            min_value=1,
            max_value=30,
            value=10,
            help="Maximum number of pages to scrape from the website"
        )
    
    with col2:
        content_option = st.radio(
            "Content to Include",
            options=["Important Pages Only", "All Pages"],
            index=0,
            help="Choose whether to include content from all pages or only important pages"
        )
    
    # Scrape button
    if st.button("Start Scraping"):
        if not company_url:
            st.error("Please enter a valid company website URL.")
            return
        
        # Save the URL to session state to remember it
        st.session_state.company_url = company_url
        
        # Initialize the scraper
        scraper = CompanyScraper(company_url, max_pages=max_pages)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define progress callback
        def update_progress(current, total):
            progress = min(current / total, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Scraping page {current} of {total}...")
        
        # Run the scraper
        with st.spinner("Scraping website content..."):
            try:
                # Run the scraper
                pages_content = asyncio.run(scraper.scrape_website(progress_callback=update_progress))
                
                # Get the combined content based on user's choice
                if content_option == "Important Pages Only":
                    combined_content = scraper.get_important_pages_content()
                else:
                    combined_content = scraper.get_combined_content()
                
                # Display results
                st.success(f"Successfully scraped {len(pages_content)} pages from {company_url}")
                
                # Show the important pages that were scraped
                if scraper.important_pages:
                    st.subheader("Important Pages Scraped")
                    for url in scraper.important_pages:
                        st.write(f"- {url}")
                
                # Show a preview of the content
                st.subheader("Content Preview")
                st.text_area(
                    "Scraped Content",
                    value=combined_content[:1000] + "..." if len(combined_content) > 1000 else combined_content,
                    height=200,
                    disabled=True
                )
                
                # Save to session state
                if "company_context" not in st.session_state:
                    st.session_state.company_context = {}
                
                # Update the company context with the scraped content
                st.session_state.company_context["website_content"] = combined_content
                
                # Also save the URL in the company context
                st.session_state.company_context["url"] = company_url
                
                # Provide option to continue
                st.info("Website content has been added to your company context. You can now close this dialog and proceed with AI analysis.")
                
            except Exception as e:
                logger.error(f"Error scraping website: {str(e)}")
                st.error(f"An error occurred while scraping the website: {str(e)}")
            finally:
                # Clean up progress indicators
                progress_bar.empty()
                status_text.empty()

async def scrape_company_website(url: str, max_pages: int = 10) -> Dict[str, str]:
    """
    Scrape a company website and return the content.
    
    Args:
        url: The company website URL
        max_pages: Maximum number of pages to scrape
        
    Returns:
        Dictionary with website content
    """
    try:
        # Initialize the scraper
        scraper = CompanyScraper(url, max_pages=max_pages)
        
        # Run the scraper
        await scraper.scrape_website()
        
        # Get the combined content
        combined_content = scraper.get_combined_content()
        
        # Return the content
        return {
            "website_content": combined_content,
            "important_pages": scraper.important_pages,
            "pages_scraped": len(scraper.pages_content)
        }
    except Exception as e:
        logger.error(f"Error scraping company website: {str(e)}")
        return {
            "website_content": f"Error scraping website: {str(e)}",
            "important_pages": [],
            "pages_scraped": 0
        } 