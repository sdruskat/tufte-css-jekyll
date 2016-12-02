# tufte-css-jekyll

## [tufte-css](https://github.com/edwardtufte/tufte-css) wrapped in a comfy Jekyll theme with `rake` support

*tufte-css-jekyll* aims at closely mimicking the [Edward Tufte](https://www.edwardtufte.com/tufte/)-inspired styles from [*tufte-css*](https://github.com/edwardtufte/tufte-css) ([MIT License](https://github.com/edwardtufte/tufte-css/blob/gh-pages/LICENSE)) in the framework of a [Jekyll](http://jekyllrb.com/) theme for satic pages and blog posts.

It is based heavily on previous work by [Clay Harmon](http://www.clayharmon.com/), who provides [*tufte-jekyll*](https://github.com/clayh53/tufte-jekyll) ([MIT License](https://github.com/clayh53/tufte-jekyll/blob/master/LICENSE)) which also draws heavily on *tufte-css*, albeit with some stylistical deviations. In comparison, *tufte-css-jekyll* tries to stay as true to *tufte-css* as possible.

*tufte-css-jekyll* also makes use of a boilerplate [`rake`](https://github.com/ruby/rake) [file](https://github.com/sdruskat/tufte-css-jekyll/blob/src/Rakefile) (provided by [Ellen Gummesson](http://ellengummesson.com/) at [jekyll-rake-boilerplate](https://github.com/gummesson/jekyll-rake-boilerplate)), which makes it easy to commandeer the Jekyll site via a number of easy-to-use [commands](#work-with-the-site-comfortably-with-rake). It alo provides a custom [Rakefile](https://github.com/sdruskat/tufte-css-jekyll/blob/src/DeployToGithub.Rakefile), which makes it easy to push the site to a [GitHub page](https://pages.github.com/).

## Demo page

A demo-page can be found at [sdruskat.github.io/tufte-css-jekyll/](https://sdruskat.github.io/tufte-css-jekyll/). The page ["Tufte CSS"](https://sdruskat.github.io/tufte-css-jekyll/page/) on this site aims at reproducing the [*tufte-css* demo page](https://edwardtufte.github.io/tufte-css/).

## Installation

[Download](https://github.com/sdruskat/tufte-css-jekyll/releases) or [clone this repository](https://github.com/sdruskat/tufte-css-jekyll.git).

## Usage

The source files live in the default branch `src`. Make your changes there, and you're ready to deploy.

If you are new to Jekyll, check out the [Jekyll documentation](https://jekyllrb.com/docs/home/) first.

### Some theme specifics

- The **large site title (and subtitle)** can be switched on/off by setting the value for `header` in `_config.yml` to false.

- The **order of pages** in the menu can be determined by defining a `weight` for the pages.

### Building and testing the site

In order to build and test the site with Jekyll's own tools, go the the root folder of the project on the command line and do

```
jekyll build
jekyll serve -w
```

Navigate to the link provided by the Jekyll CLI and you will see the newly built page.

### Work with the site comfortably with `rake`

From the root of the project, you can run a number of commands in order to work with the site. They are basically those from [Ellen Gummesson](http://ellengummesson.com/)'s boilerplate' [`rake` file](https://github.com/gummesson/jekyll-rake-boilerplate). They are listed here for convenience.

- `rake post["Title"]` creates a new post in the `_posts` directory by reading the default template file, adding the title you've specified and generating a filename by using the current date and the title.

- `rake draft["Title"]` creates a new post in the `_drafts` directory by reading the default template file, adding the title you've specified and generating a filename.

- `rake publish` moves a post from the `_drafts` directory to the `_posts` directory and appends the current date to it. It'll list all drafts and then you'll get to choose which draft to move by providing a number.

- `rake page["Title","path/to/folder"]` creates a new page. If the file path is not specified the page will get placed in the site's source directory.

### Deploying to [GitHub pages](https://pages.github.com/)

Depending on whether you want to deploy to a project (branch `gh-pages`) or a user/organization (branch master in specific repository `user.github.io`), you can use the [`DeployToGithub.Rakefile`](https://github.com/sdruskat/tufte-css-jekyll/blob/src/DeployToGithub.Rakefile) with the respective argument.

`rake -f DeployToGithub.Rakefile publish` will publish the page to the `gh-pages` branch of your repository, while `rake -f DeployToGithub.Rakefile publishmaster` will publish a user/organization page.

In detail, this will `git commit` changes to `src` with a boilerplate commit message and `git push` to `origin/src`, then checkout `gh-pages` (or `master`), remove everything, copy the build result (in `./site`) from a tmp directory to the branch, `commit` with a timestamp, force `push` to the respective branch, an checkout `src` again.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sdruskat/tufte-css-jekyll. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The theme is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

