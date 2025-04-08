# In the ChromaVectorstore class, modify the scrape_with_playwright method to improve chunking

async def scrape_with_playwright(self, urls):
    """Scrape content from URLs using Playwright"""
    results = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        
        for doc in urls:
            url = doc["url"]
            title = doc["title"]
            print(f"Processing with Playwright: {url}")
            
            try:
                page = await context.new_page()
                
                # Navigate with timeout and wait until network is idle
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Wait a bit more for any delayed rendering
                await page.wait_for_timeout(2000)
                
                # Try getting the page title
                page_title = await page.title()
                if not page_title:
                    page_title = title
                
                # Extract content - try multiple selectors
                selectors = [
                    "main", 
                    "article", 
                    ".content",
                    ".prose",
                    ".docusaurus-content",
                    "[role='main']",
                    ".container main",
                    ".markdown"
                ]
                
                content_html = None
                for selector in selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            content_html = await element.inner_html()
                            print(f"Found content using selector: {selector}")
                            break
                    except Exception as e:
                        continue
                
                # If no selector worked, capture the entire body content
                if not content_html:
                    print(f"No specific content container found for {url}, using body")
                    body = await page.query_selector("body")
                    if body:
                        content_html = await body.inner_html()
                    
                if content_html:
                    # Use BeautifulSoup to parse the extracted HTML
                    soup = BeautifulSoup(content_html, 'html.parser')
                    
                    # Find all headings to structure the content
                    headings = soup.find_all(['h1', 'h2', 'h3'])  # Reduced to major headings for larger chunks
                    
                    # Initialize sections
                    sections = []
                    if headings:
                        # Process content by headings
                        for i, heading in enumerate(headings):
                            # Get heading text
                            heading_text = heading.get_text(strip=True)
                            
                            # Find all elements between this heading and the next one
                            content_elements = []
                            current = heading.next_sibling
                            
                            # Get next heading for boundary
                            next_heading = headings[i+1] if i < len(headings)-1 else None
                            
                            while current and (not next_heading or current != next_heading):
                                if current.name in ['p', 'ul', 'ol', 'pre', 'div', 'table', 'h4', 'h5', 'h6']:
                                    content = current.get_text(strip=True)
                                    if content:
                                        content_elements.append(content)
                                current = current.next_sibling
                            
                            # Add section if it has content
                            if content_elements:
                                sections.append({
                                    "header": heading_text,
                                    "content": content_elements
                                })
                    
                    # If no structure was found, use a fallback approach
                    if not sections:
                        # Extract all paragraphs and list items
                        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
                        list_items = []
                        for ul in soup.find_all(['ul', 'ol']):
                            for li in ul.find_all('li'):
                                text = li.get_text(strip=True)
                                if text:
                                    list_items.append(f"• {text}")
                        
                        all_content = paragraphs + list_items
                        if all_content:
                            # Combine into larger chunks of approximately 1000 characters
                            combined_content = []
                            current_chunk = []
                            current_length = 0
                            
                            for item in all_content:
                                if current_length + len(item) > 1000:
                                    if current_chunk:  # Only append if we have content
                                        combined_content.append(current_chunk)
                                    current_chunk = [item]
                                    current_length = len(item)
                                else:
                                    current_chunk.append(item)
                                    current_length += len(item)
                            
                            if current_chunk:  # Add the last chunk if it exists
                                combined_content.append(current_chunk)
                            
                            # Create sections from the combined content
                            for i, chunk in enumerate(combined_content):
                                sections.append({
                                    "header": f"{page_title} - Part {i+1}",
                                    "content": chunk
                                })
                    
                    # Save results - create larger, more meaningful chunks
                    if sections:
                        for i, section in enumerate(sections):
                            # Create unique ID for the chunk
                            parsed_url = urlparse(url)
                            path_parts = parsed_url.path.strip('/').split('/')
                            path_id = '-'.join(path_parts) if path_parts else 'root'
                            chunk_id = f"{path_id}-{i}"
                            
                            # Join content with newlines and ensure proper spacing
                            section_content = "\n\n".join(section["content"])
                            
                            # Make sure the URL is correct
                            clean_url = url
                            if "Please generate document codes" in clean_url:
                                clean_url = clean_url.replace("Please generate document codes", "")
                            
                            results.append({
                                "id": chunk_id,
                                "title": f"{page_title} - {section['header']}" if section['header'] != page_title else page_title,
                                "text": f"{section['header']}\n\n{section_content}",
                                "url": clean_url,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                            print(f"Extracted section: {section['header']} ({len(section_content)} chars)")
                    else:
                        print(f"⚠️ No content sections found for {url}")
                
                else:
                    print(f"⚠️ No content HTML found for {url}")
                
                # Close the page to free resources
                await page.close()
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error processing {url}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clean up
        await context.close()
        await browser.close()
    
    return results